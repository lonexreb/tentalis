"""Trainer protocol, TrainStepResult, MockTrainer, and GRPOTrainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from src.events.types import TrainingRolloutEvent

logger = logging.getLogger(__name__)


class TrainStepResult(BaseModel):
    loss: float
    mean_advantage: float
    std_advantage: float
    checkpoint_path: str | None = None
    step_count: int = 0


@runtime_checkable
class Trainer(Protocol):
    async def train_step(self, batch: list[TrainingRolloutEvent]) -> TrainStepResult: ...
    def checkpoint_path(self) -> str | None: ...


class MockTrainer:
    """No-op trainer for testing without torch."""

    def __init__(self) -> None:
        self._step_count = 0

    async def train_step(self, batch: list[TrainingRolloutEvent]) -> TrainStepResult:
        self._step_count += 1
        scores = [r.outcome_score for r in batch]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        logger.info(
            "MockTrainer step %d: batch_size=%d, mean_score=%.3f",
            self._step_count, len(batch), mean_score,
        )
        return TrainStepResult(
            loss=1.0 / self._step_count,
            mean_advantage=0.0,
            std_advantage=1.0,
            step_count=self._step_count,
        )

    def checkpoint_path(self) -> str | None:
        return None


class GRPOTrainer:
    """GRPO trainer with LoRA fine-tuning on a causal LM."""

    def __init__(
        self,
        model_name: str = "distilgpt2",
        lr: float = 1e-4,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.1,
        checkpoint_dir: str = "checkpoints",
        lora_rank: int = 8,
        checkpoint_every: int = 5,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._lr = lr
        self._clip_epsilon = clip_epsilon
        self._kl_beta = kl_beta
        self._checkpoint_dir = Path(checkpoint_dir)
        self._lora_rank = lora_rank
        self._checkpoint_every = checkpoint_every
        self._device = device
        self._step_count = 0
        self._last_checkpoint: str | None = None

        self._model = None
        self._ref_model = None
        self._tokenizer = None
        self._optimizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        logger.info("Loading model %s for GRPO training", self._model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(self._model_name)
        base_model.to(self._device)

        # Reference model (frozen) for KL penalty
        import copy
        self._ref_model = copy.deepcopy(base_model)
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad = False

        # LoRA adapter on the trainable model
        lora_config = LoraConfig(
            r=self._lora_rank,
            lora_alpha=self._lora_rank * 2,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(base_model, lora_config)
        self._model.train()

        import torch
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._lr
        )

        logger.info("Model loaded: %s (LoRA r=%d)", self._model_name, self._lora_rank)

    async def train_step(self, batch: list[TrainingRolloutEvent]) -> TrainStepResult:
        import torch
        from src.training.grpo import compute_group_advantages, clipped_surrogate_loss, kl_penalty

        self._ensure_loaded()
        self._step_count += 1

        # Group rollouts by prompt and compute advantages
        groups: dict[str, list[TrainingRolloutEvent]] = {}
        for r in batch:
            groups.setdefault(r.prompt, []).append(r)

        all_advantages: list[float] = []
        all_texts: list[str] = []
        for prompt, rollouts in groups.items():
            rewards = [r.outcome_score for r in rollouts]
            advs = compute_group_advantages(rewards)
            all_advantages.extend(advs)
            all_texts.extend(
                f"{prompt}\n{r.response}" for r in rollouts
            )

        if not all_texts:
            return TrainStepResult(
                loss=0.0, mean_advantage=0.0, std_advantage=0.0,
                step_count=self._step_count,
            )

        # Tokenize
        encodings = self._tokenizer(
            all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._device)

        # Forward pass — policy model
        outputs = self._model(**encodings, labels=encodings["input_ids"])
        log_probs = -outputs.loss  # avg negative log likelihood

        # Forward pass — reference model
        with torch.no_grad():
            ref_outputs = self._ref_model(**encodings, labels=encodings["input_ids"])
            ref_log_probs = -ref_outputs.loss

        # Compute loss components
        advantages_t = torch.tensor(all_advantages, dtype=torch.float32, device=self._device)
        ratios = torch.exp(log_probs - ref_log_probs).expand_as(advantages_t)
        clip_loss = clipped_surrogate_loss(ratios, advantages_t, self._clip_epsilon)
        kl_loss = kl_penalty(log_probs.expand_as(ref_log_probs), ref_log_probs, self._kl_beta)
        total_loss = clip_loss + kl_loss

        # Backward + optimize
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        loss_val = total_loss.item()
        mean_adv = sum(all_advantages) / len(all_advantages)
        std_adv = (
            sum((a - mean_adv) ** 2 for a in all_advantages) / len(all_advantages)
        ) ** 0.5

        # Periodic checkpointing
        checkpoint_path = None
        if self._step_count % self._checkpoint_every == 0:
            checkpoint_path = self._save_checkpoint()

        logger.info(
            "GRPOTrainer step %d: loss=%.4f, mean_adv=%.4f, std_adv=%.4f",
            self._step_count, loss_val, mean_adv, std_adv,
        )

        return TrainStepResult(
            loss=loss_val,
            mean_advantage=mean_adv,
            std_advantage=std_adv,
            checkpoint_path=checkpoint_path,
            step_count=self._step_count,
        )

    def _save_checkpoint(self) -> str:
        path = self._checkpoint_dir / f"v{self._step_count:04d}"
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path))
        self._tokenizer.save_pretrained(str(path))
        self._last_checkpoint = str(path)
        logger.info("Saved checkpoint to %s", path)
        return str(path)

    def checkpoint_path(self) -> str | None:
        return self._last_checkpoint


class DAPOTrainer:
    """Full DAPO trainer: dynamic sampling + asymmetric clipping + entropy bonus.

    Extends GRPOTrainer's approach with DAPO-specific loss and pre-filtering.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        lr: float = 1e-4,
        clip_epsilon: float = 0.2,
        clip_epsilon_high: float = 0.28,
        kl_beta: float = 0.1,
        beta_entropy: float = 0.01,
        min_reward_threshold: float = 0.1,
        checkpoint_dir: str = "checkpoints",
        lora_rank: int = 8,
        checkpoint_every: int = 5,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._lr = lr
        self._clip_epsilon = clip_epsilon
        self._clip_epsilon_high = clip_epsilon_high
        self._kl_beta = kl_beta
        self._beta_entropy = beta_entropy
        self._min_reward_threshold = min_reward_threshold
        self._checkpoint_dir = checkpoint_dir
        self._lora_rank = lora_rank
        self._checkpoint_every = checkpoint_every
        self._device = device
        self._step_count = 0
        self._model = None
        self._ref_model = None
        self._tokenizer = None
        self._optimizer = None
        self._last_checkpoint: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import copy

        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._ref_model = copy.deepcopy(base).eval()
        for p in self._ref_model.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=self._lora_rank,
            lora_alpha=2 * self._lora_rank,
            target_modules=["c_attn"],
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(base, lora_cfg)
        self._model.to(self._device)
        self._ref_model.to(self._device)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._lr,
        )

    async def train_step(
        self, batch: list[TrainingRolloutEvent],
    ) -> TrainStepResult:
        import torch

        from src.training.dapo import dapo_loss, dynamic_sample_filter
        from src.training.grpo import compute_group_advantages, kl_penalty

        self._ensure_loaded()
        self._step_count += 1

        # Dynamic sampling filter
        filtered = dynamic_sample_filter(
            batch, min_reward_threshold=self._min_reward_threshold,
        )
        if not filtered:
            return TrainStepResult(
                loss=0.0, mean_advantage=0.0, std_advantage=0.0,
                step_count=self._step_count,
            )

        # Group by prompt, compute advantages
        groups: dict[str, list[TrainingRolloutEvent]] = {}
        for r in filtered:
            groups.setdefault(r.prompt, []).append(r)

        all_advantages: list[float] = []
        texts: list[str] = []
        for prompt, rollouts in groups.items():
            rewards = [r.outcome_score for r in rollouts]
            advs = compute_group_advantages(rewards)
            all_advantages.extend(advs)
            texts.extend(f"{prompt}\n{r.response}" for r in rollouts)

        enc = self._tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(self._device)

        # Reference forward pass
        with torch.no_grad():
            ref_out = self._ref_model(**enc)
            ref_log_probs = ref_out.logits.log_softmax(-1).mean(dim=(1, 2))

        # Policy forward pass
        out = self._model(**enc)
        log_probs = out.logits.log_softmax(-1).mean(dim=(1, 2))

        advantages = torch.tensor(all_advantages, device=self._device)
        ratios = torch.exp(log_probs - ref_log_probs.detach())

        # DAPO loss (asymmetric clip + entropy bonus)
        loss = dapo_loss(
            ratios, advantages, out.logits,
            clip_eps=self._clip_epsilon,
            clip_eps_high=self._clip_epsilon_high,
            beta_entropy=self._beta_entropy,
        )
        kl_loss = kl_penalty(log_probs, ref_log_probs.detach(), self._kl_beta)
        total_loss = loss + kl_loss

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        # Checkpoint
        ckpt_path = None
        if self._step_count % self._checkpoint_every == 0:
            ckpt_path = self._save_checkpoint()

        mean_adv = sum(all_advantages) / len(all_advantages) if all_advantages else 0.0
        std_adv = (
            (sum((a - mean_adv) ** 2 for a in all_advantages) / len(all_advantages)) ** 0.5
            if all_advantages else 0.0
        )

        return TrainStepResult(
            loss=total_loss.item(),
            mean_advantage=mean_adv,
            std_advantage=std_adv,
            checkpoint_path=ckpt_path,
            step_count=self._step_count,
        )

    def _save_checkpoint(self) -> str:
        from pathlib import Path

        path = str(Path(self._checkpoint_dir) / f"dapo-v{self._step_count:04d}")
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        self._last_checkpoint = path
        return path

    def checkpoint_path(self) -> str | None:
        return self._last_checkpoint
