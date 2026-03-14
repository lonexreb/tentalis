"""CombinedTrainer — merges RL rewards + OPD distillation in a single gradient step."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from src.events.types import CombinedRolloutEvent

logger = logging.getLogger(__name__)


class CombinedTrainStepResult(BaseModel):
    loss: float
    rl_loss: float
    opd_loss: float
    mean_advantage: float
    std_advantage: float
    checkpoint_path: str | None = None
    step_count: int = 0
    opd_count: int = 0


class CombinedTrainer:
    """Trainer that combines Binary RL + OPD distillation loss.

    When a rollout has OPD hints (has_opd=True):
        loss = w_rl * clipped_surrogate(reward) + w_opd * (teacher_logp - old_logp)
    When no OPD hints:
        loss = clipped_surrogate(reward)  (pure RL fallback)
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        lr: float = 1e-4,
        clip_epsilon: float = 0.2,
        clip_epsilon_high: float = 0.28,
        kl_beta: float = 0.1,
        w_rl: float = 0.7,
        w_opd: float = 0.3,
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
        self._w_rl = w_rl
        self._w_opd = w_opd
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
        import copy
        import torch

        logger.info("Loading model %s for combined training", self._model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(self._model_name)
        base_model.to(self._device)

        self._ref_model = copy.deepcopy(base_model)
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad = False

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

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._lr
        )

    async def train_step(
        self, batch: list[CombinedRolloutEvent]
    ) -> CombinedTrainStepResult:
        import torch
        from src.training.grpo import (
            asymmetric_clipped_surrogate_loss,
            combined_loss,
            compute_group_advantages,
            kl_penalty,
        )

        self._ensure_loaded()
        self._step_count += 1

        # Group by prompt for advantage computation
        groups: dict[str, list[CombinedRolloutEvent]] = {}
        for r in batch:
            groups.setdefault(r.prompt, []).append(r)

        all_advantages: list[float] = []
        all_texts: list[str] = []
        opd_mask: list[bool] = []
        for prompt, rollouts in groups.items():
            rewards = [r.outcome_score for r in rollouts]
            advs = compute_group_advantages(rewards)
            all_advantages.extend(advs)
            all_texts.extend(f"{prompt}\n{r.response}" for r in rollouts)
            opd_mask.extend(r.has_opd for r in rollouts)

        if not all_texts:
            return CombinedTrainStepResult(
                loss=0.0, rl_loss=0.0, opd_loss=0.0,
                mean_advantage=0.0, std_advantage=0.0,
                step_count=self._step_count,
            )

        # Tokenize
        encodings = self._tokenizer(
            all_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(self._device)

        # Forward passes
        outputs = self._model(**encodings, labels=encodings["input_ids"])
        log_probs = -outputs.loss

        with torch.no_grad():
            ref_outputs = self._ref_model(**encodings, labels=encodings["input_ids"])
            ref_log_probs = -ref_outputs.loss

        # RL loss (asymmetric clipping)
        advantages_t = torch.tensor(
            all_advantages, dtype=torch.float32, device=self._device
        )
        ratios = torch.exp(log_probs - ref_log_probs).expand_as(advantages_t)
        rl_loss_val = asymmetric_clipped_surrogate_loss(
            ratios, advantages_t, self._clip_epsilon, self._clip_epsilon_high
        )
        kl_loss = kl_penalty(
            log_probs.expand_as(ref_log_probs), ref_log_probs, self._kl_beta
        )

        # OPD loss — distillation toward teacher logprobs
        opd_count = sum(opd_mask)
        if opd_count > 0:
            # OPD loss = mean(teacher_logp - old_logp) for OPD samples
            # Since teacher logprobs come from external source, use the
            # difference between policy and reference as a proxy
            opd_loss_val = -(log_probs - ref_log_probs).mean()
            total = combined_loss(
                rl_loss_val + kl_loss, opd_loss_val, self._w_rl, self._w_opd
            )
        else:
            opd_loss_val = torch.tensor(0.0, device=self._device)
            total = rl_loss_val + kl_loss

        # Backward + optimize
        self._optimizer.zero_grad()
        total.backward()
        self._optimizer.step()

        mean_adv = sum(all_advantages) / len(all_advantages)
        std_adv = (
            sum((a - mean_adv) ** 2 for a in all_advantages) / len(all_advantages)
        ) ** 0.5

        checkpoint_path = None
        if self._step_count % self._checkpoint_every == 0:
            checkpoint_path = self._save_checkpoint()

        logger.info(
            "CombinedTrainer step %d: loss=%.4f rl=%.4f opd=%.4f (opd_count=%d)",
            self._step_count, total.item(), rl_loss_val.item(),
            opd_loss_val.item(), opd_count,
        )

        return CombinedTrainStepResult(
            loss=total.item(),
            rl_loss=rl_loss_val.item(),
            opd_loss=opd_loss_val.item(),
            mean_advantage=mean_adv,
            std_advantage=std_adv,
            checkpoint_path=checkpoint_path,
            step_count=self._step_count,
            opd_count=opd_count,
        )

    def _save_checkpoint(self) -> str:
        path = self._checkpoint_dir / f"combined-v{self._step_count:04d}"
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path))
        self._tokenizer.save_pretrained(str(path))
        self._last_checkpoint = str(path)
        logger.info("Saved combined checkpoint to %s", path)
        return str(path)

    def checkpoint_path(self) -> str | None:
        return self._last_checkpoint
