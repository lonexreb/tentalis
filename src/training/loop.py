"""TrainingLoop orchestrator — wires bridge + trainer + ModelUpdateEvent publishing."""

from __future__ import annotations

import logging
from pathlib import Path

from src.events.bus import EventBus
from src.events.topics import MODEL_UPDATES, TRAINING_COMBINED
from src.events.types import (
    CombinedRolloutEvent,
    ModelUpdateEvent,
    TrainingRolloutEvent,
)
from src.training.bridge import NATSTrainingBridge, RolloutBuffer
from src.training.trainer import Trainer, TrainStepResult

logger = logging.getLogger(__name__)


class TrainingLoop:
    """Orchestrates the training pipeline: bridge → trainer → model updates.

    Supports both standard TrainingRolloutEvent (via NATSTrainingBridge) and
    CombinedRolloutEvent (via direct NATS subscription) when combined training
    is enabled.
    """

    def __init__(
        self,
        bus: EventBus,
        trainer: Trainer,
        buffer: RolloutBuffer | None = None,
        save_path: Path | None = None,
        combined_trainer: object | None = None,
    ) -> None:
        self._bus = bus
        self._trainer = trainer
        self._buffer = buffer or RolloutBuffer(batch_size=8, group_size=1)
        self._save_path = save_path
        self._bridge: NATSTrainingBridge | None = None
        self._combined_trainer = combined_trainer
        self._combined_buffer: list[CombinedRolloutEvent] = []
        self._combined_batch_size = self._buffer.batch_size

    async def start(self) -> None:
        self._bridge = NATSTrainingBridge(
            self._bus,
            self._buffer,
            self._on_batch,
            save_path=self._save_path,
        )
        await self._bridge.start()

        # Subscribe to combined rollouts if a combined trainer is provided
        if self._combined_trainer is not None:
            await self._bus.subscribe(
                TRAINING_COMBINED,
                CombinedRolloutEvent,
                self._handle_combined_rollout,
            )
            logger.info("TrainingLoop started (combined training enabled)")
        else:
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

    async def _handle_combined_rollout(
        self, rollout: CombinedRolloutEvent
    ) -> None:
        self._combined_buffer.append(rollout)
        if len(self._combined_buffer) >= self._combined_batch_size:
            batch = self._combined_buffer[: self._combined_batch_size]
            self._combined_buffer = self._combined_buffer[self._combined_batch_size :]
            await self._on_combined_batch(batch)

    async def _on_combined_batch(
        self, batch: list[CombinedRolloutEvent]
    ) -> None:
        logger.info("Combined training on batch of %d rollouts", len(batch))
        result = await self._combined_trainer.train_step(batch)

        logger.info(
            "Combined train step %d: loss=%.4f rl=%.4f opd=%.4f",
            result.step_count, result.loss, result.rl_loss, result.opd_loss,
        )

        if result.checkpoint_path:
            # Include target_worker_id if all rollouts are from the same worker
            worker_ids = {r.worker_id for r in batch}
            target = worker_ids.pop() if len(worker_ids) == 1 else None

            event = ModelUpdateEvent(
                model_version=f"combined-v{result.step_count:04d}",
                checkpoint_path=result.checkpoint_path,
                target_worker_id=target,
                metrics={
                    "loss": result.loss,
                    "rl_loss": result.rl_loss,
                    "opd_loss": result.opd_loss,
                    "mean_advantage": result.mean_advantage,
                    "opd_count": result.opd_count,
                },
            )
            await self._bus.publish(MODEL_UPDATES, event)
            logger.info("Published combined ModelUpdateEvent: %s", event.model_version)
