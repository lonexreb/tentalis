"""TinkerBackend — Trainer protocol adapter for Tinker managed cloud training."""

from __future__ import annotations

import logging

from src.events.types import TrainingRolloutEvent
from src.training.trainer import TrainStepResult

logger = logging.getLogger(__name__)


class TinkerBackend:
    """Implements the Trainer protocol using the Tinker SDK for cloud-managed RL training.

    Tinker handles forward/backward passes and optimizer steps in their cloud,
    so no local GPU is required.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "https://api.tinker.thinkingmachines.ai",
        model_name: str = "qwen2.5:1.5b",
        learning_rate: float = 1e-4,
        clip_range: float = 0.2,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model_name = model_name
        self._learning_rate = learning_rate
        self._clip_range = clip_range
        self._step_count = 0
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import tinker

            self._client = tinker.Client(
                api_key=self._api_key,
                base_url=self._base_url,
            )
            logger.info("Connected to Tinker at %s", self._base_url)
        except ImportError:
            raise ImportError(
                "tinker package not installed. "
                'Install with: pip install -e ".[tinker]"'
            )

    async def train_step(self, batch: list[TrainingRolloutEvent]) -> TrainStepResult:
        self._ensure_client()
        self._step_count += 1

        # Prepare rollout data for Tinker API
        samples = []
        for rollout in batch:
            samples.append({
                "prompt": rollout.prompt,
                "response": rollout.response,
                "reward": rollout.outcome_score,
                "step_scores": rollout.step_scores,
            })

        # Tinker SDK: forward_backward() computes gradients in the cloud
        fb_result = self._client.forward_backward(
            model=self._model_name,
            samples=samples,
            clip_range=self._clip_range,
        )

        # Tinker SDK: optim_step() applies the gradient update
        optim_result = self._client.optim_step(
            model=self._model_name,
            learning_rate=self._learning_rate,
        )

        loss = fb_result.get("loss", 0.0) if isinstance(fb_result, dict) else 0.0
        mean_adv = fb_result.get("mean_advantage", 0.0) if isinstance(fb_result, dict) else 0.0
        std_adv = fb_result.get("std_advantage", 0.0) if isinstance(fb_result, dict) else 0.0
        checkpoint = optim_result.get("checkpoint_id", None) if isinstance(optim_result, dict) else None

        logger.info(
            "TinkerBackend step %d: loss=%.4f, mean_adv=%.4f",
            self._step_count, loss, mean_adv,
        )

        return TrainStepResult(
            loss=loss,
            mean_advantage=mean_adv,
            std_advantage=std_adv,
            checkpoint_path=checkpoint,
            step_count=self._step_count,
        )

    def checkpoint_path(self) -> str | None:
        return None
