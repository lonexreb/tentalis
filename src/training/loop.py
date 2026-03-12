"""TrainingLoop orchestrator — wires bridge + trainer + ModelUpdateEvent publishing."""

from __future__ import annotations

import logging
from pathlib import Path

from src.events.bus import EventBus
from src.events.topics import MODEL_UPDATES
from src.events.types import ModelUpdateEvent, TrainingRolloutEvent
from src.training.bridge import NATSTrainingBridge, RolloutBuffer
from src.training.trainer import Trainer, TrainStepResult

logger = logging.getLogger(__name__)


class TrainingLoop:
    """Orchestrates the training pipeline: bridge → trainer → model updates."""

    def __init__(
        self,
        bus: EventBus,
        trainer: Trainer,
        buffer: RolloutBuffer | None = None,
        save_path: Path | None = None,
    ) -> None:
        self._bus = bus
        self._trainer = trainer
        self._buffer = buffer or RolloutBuffer(batch_size=8, group_size=1)
        self._save_path = save_path
        self._bridge: NATSTrainingBridge | None = None

    async def start(self) -> None:
        self._bridge = NATSTrainingBridge(
            self._bus,
            self._buffer,
            self._on_batch,
            save_path=self._save_path,
        )
        await self._bridge.start()
        logger.info("TrainingLoop started")

    async def _on_batch(self, batch: list[TrainingRolloutEvent]) -> None:
        logger.info("Training on batch of %d rollouts", len(batch))
        result: TrainStepResult = await self._trainer.train_step(batch)

        logger.info(
            "Train step %d: loss=%.4f, mean_adv=%.4f",
            result.step_count, result.loss, result.mean_advantage,
        )

        if result.checkpoint_path:
            event = ModelUpdateEvent(
                model_version=f"v{result.step_count:04d}",
                checkpoint_path=result.checkpoint_path,
                metrics={
                    "loss": result.loss,
                    "mean_advantage": result.mean_advantage,
                    "std_advantage": result.std_advantage,
                },
            )
            await self._bus.publish(MODEL_UPDATES, event)
            logger.info("Published ModelUpdateEvent: %s", event.model_version)
