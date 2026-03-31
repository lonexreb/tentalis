"""CISPOTrainer — GRPO + contrastive trajectory loss."""

from __future__ import annotations

import logging

from pydantic import BaseModel

from src.events.types import TrainingRolloutEvent
from src.training.cispo import ContrastivePair, build_contrastive_pairs, contrastive_trajectory_loss
from src.training.grpo import compute_group_advantages, multi_loss

logger = logging.getLogger(__name__)


class CISPOTrainStepResult(BaseModel):
    loss: float
    rl_loss: float
    contrastive_loss: float
    mean_advantage: float
    std_advantage: float
    checkpoint_path: str | None = None
    step_count: int = 0
    num_contrastive_pairs: int = 0


class CISPOTrainer:
    """GRPO trainer augmented with contrastive trajectory loss.

    total_loss = w_rl * rl_loss + w_contrastive * contrastive_loss

    When no contrastive pairs are available, falls back to pure RL loss.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        lr: float = 1e-4,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.1,
        w_rl: float = 0.7,
        w_contrastive: float = 0.3,
        contrastive_margin: float = 0.5,
        checkpoint_dir: str = "checkpoints",
        lora_rank: int = 8,
        checkpoint_every: int = 5,
        device: str = "cpu",
        trajectory_store: object | None = None,
    ) -> None:
        self._model_name = model_name
        self._lr = lr
        self._clip_epsilon = clip_epsilon
        self._kl_beta = kl_beta
        self._w_rl = w_rl
        self._w_contrastive = w_contrastive
        self._contrastive_margin = contrastive_margin
        self._checkpoint_dir = checkpoint_dir
        self._lora_rank = lora_rank
        self._checkpoint_every = checkpoint_every
        self._device = device
        self._trajectory_store = trajectory_store
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
    ) -> CISPOTrainStepResult:
        import torch

        from src.training.grpo import clipped_surrogate_loss, kl_penalty

        self._ensure_loaded()
        self._step_count += 1

        # --- RL loss (standard GRPO) ---
        groups: dict[str, list[TrainingRolloutEvent]] = {}
        for r in batch:
            groups.setdefault(r.prompt, []).append(r)

        all_advantages: list[float] = []
        texts: list[str] = []
        for prompt, rollouts in groups.items():
            rewards = [r.outcome_score for r in rollouts]
            advs = compute_group_advantages(rewards)
            all_advantages.extend(advs)
            texts.extend(f"{prompt}\n{r.response}" for r in rollouts)

        if not texts:
            return CISPOTrainStepResult(
                loss=0.0, rl_loss=0.0, contrastive_loss=0.0,
                mean_advantage=0.0, std_advantage=0.0,
                step_count=self._step_count,
            )

        enc = self._tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(self._device)

        with torch.no_grad():
            ref_out = self._ref_model(**enc)
            ref_log_probs = ref_out.logits.log_softmax(-1).mean(dim=(1, 2))

        out = self._model(**enc)
        log_probs = out.logits.log_softmax(-1).mean(dim=(1, 2))

        advantages = torch.tensor(all_advantages, device=self._device)
        ratios = torch.exp(log_probs - ref_log_probs.detach())

        rl_loss = clipped_surrogate_loss(ratios, advantages, self._clip_epsilon)
        kl_loss = kl_penalty(log_probs, ref_log_probs.detach(), self._kl_beta)
        rl_total = rl_loss + kl_loss

        # --- Contrastive loss ---
        contrastive_loss_val = torch.tensor(0.0, device=self._device)
        num_pairs = 0

        if self._trajectory_store is not None:
            aligned = self._trajectory_store.query(min_score=0.7, limit=50)
            misaligned = self._trajectory_store.query(max_score=0.3, limit=50)
            pairs = build_contrastive_pairs(aligned, misaligned)
            num_pairs = len(pairs)

            if pairs:
                pos_scores = torch.tensor(
                    [p.pos_score for p in pairs], device=self._device,
                )
                neg_scores = torch.tensor(
                    [p.neg_score for p in pairs], device=self._device,
                )
                contrastive_loss_val = contrastive_trajectory_loss(
                    pos_scores, neg_scores, margin=self._contrastive_margin,
                )

        # --- Combined loss ---
        total = multi_loss(
            {"rl": rl_total, "contrastive": contrastive_loss_val},
            {"rl": self._w_rl, "contrastive": self._w_contrastive},
        )

        self._optimizer.zero_grad()
        total.backward()
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

        return CISPOTrainStepResult(
            loss=total.item(),
            rl_loss=rl_total.item(),
            contrastive_loss=contrastive_loss_val.item(),
            mean_advantage=mean_adv,
            std_advantage=std_adv,
            checkpoint_path=ckpt_path,
            step_count=self._step_count,
            num_contrastive_pairs=num_pairs,
        )

    def _save_checkpoint(self) -> str:
        from pathlib import Path

        path = str(Path(self._checkpoint_dir) / f"cispo-v{self._step_count:04d}")
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        self._last_checkpoint = path
        logger.info("CISPO checkpoint saved: %s", path)
        return path

    def checkpoint_path(self) -> str | None:
        return self._last_checkpoint
