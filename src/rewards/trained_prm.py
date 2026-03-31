"""Trained Process Reward Model — frozen LLM + learned scalar reward head."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RewardHead:
    """Scalar reward head on top of a frozen language model.

    Architecture: Linear(hidden_dim, 256) -> ReLU -> Linear(256, 1) -> Sigmoid
    """

    def __init__(self, hidden_dim: int) -> None:
        import torch.nn as nn

        self._head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def __call__(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        return self._head(hidden_states)

    def parameters(self):
        return self._head.parameters()

    def to(self, device: str) -> "RewardHead":
        self._head.to(device)
        return self

    def state_dict(self):
        return self._head.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._head.load_state_dict(state_dict)

    def train(self) -> None:
        self._head.train()

    def eval(self) -> None:
        self._head.eval()


class TrainedPRM:
    """Trained Process Reward Model.

    Wraps a frozen base LLM + learned RewardHead. Only the reward head is
    trained; the base model provides embeddings.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cpu",
        checkpoint_path: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._checkpoint_path = checkpoint_path
        self._base_model = None
        self._tokenizer = None
        self._reward_head: RewardHead | None = None

    def _ensure_loaded(self) -> None:
        if self._base_model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._base_model = AutoModelForCausalLM.from_pretrained(
            self._model_name, output_hidden_states=True,
        )
        self._base_model.eval()
        for p in self._base_model.parameters():
            p.requires_grad = False
        self._base_model.to(self._device)

        hidden_dim = self._base_model.config.hidden_size
        self._reward_head = RewardHead(hidden_dim).to(self._device)

        if self._checkpoint_path:
            self.load(self._checkpoint_path)

    def predict(self, prompt: str, step: str) -> float:
        """Score a single step given the prompt context. Returns 0.0-1.0."""
        return self.predict_batch([prompt], [step])[0]

    def predict_batch(self, prompts: list[str], steps: list[str]) -> list[float]:
        """Batch prediction for efficiency."""
        import torch

        self._ensure_loaded()

        texts = [f"{p}\n{s}" for p, s in zip(prompts, steps)]
        enc = self._tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._base_model(**enc)
            # Use last hidden state, last token position
            hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)
            # Get last non-padding token for each sequence
            lengths = enc.attention_mask.sum(dim=1) - 1  # (batch,)
            last_hidden = hidden[
                torch.arange(hidden.size(0)), lengths
            ]  # (batch, hidden)

        self._reward_head.eval()
        with torch.no_grad():
            scores = self._reward_head(last_hidden).squeeze(-1)  # (batch,)

        return scores.cpu().tolist()

    def save(self, path: str) -> None:
        """Save only the reward head weights (base model is frozen)."""
        import torch

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._reward_head.state_dict(), save_dir / "reward_head.pt")
        logger.info("Saved PRM reward head to %s", path)

    def load(self, path: str) -> None:
        """Load reward head weights."""
        import torch

        load_path = Path(path) / "reward_head.pt"
        if load_path.exists():
            state = torch.load(load_path, map_location=self._device, weights_only=True)
            self._reward_head.load_state_dict(state)
            logger.info("Loaded PRM reward head from %s", path)


class TrainedPRMScorer:
    """StepScorer protocol adapter for TrainedPRM.

    Runs model inference in an executor to avoid blocking the event loop.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cpu",
        checkpoint_path: str | None = None,
    ) -> None:
        self._prm = TrainedPRM(
            model_name=model_name,
            device=device,
            checkpoint_path=checkpoint_path,
        )

    async def score_steps(self, prompt: str, steps: list[str]) -> list[float]:
        """Score each step using the trained PRM."""
        loop = asyncio.get_event_loop()

        # Build progressive context (same as LLMJudgeScorer)
        prompts: list[str] = []
        step_texts: list[str] = []
        context = prompt
        for step in steps:
            prompts.append(context)
            step_texts.append(step)
            context = f"{context}\n{step}"

        scores = await loop.run_in_executor(
            None, self._prm.predict_batch, prompts, step_texts,
        )
        return scores
