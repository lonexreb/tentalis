from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from src.events.bus import EventBus
from src.events.topics import TRAINING_ROLLOUTS
from src.events.types import TrainingRolloutEvent

logger = logging.getLogger(__name__)


class RolloutBuffer:
    def __init__(self, batch_size: int = 8, group_size: int = 1) -> None:
        self.batch_size = batch_size
        self.group_size = group_size
        self._groups: dict[str, list[TrainingRolloutEvent]] = defaultdict(list)
        self._complete: list[TrainingRolloutEvent] = []

    def add(self, rollout: TrainingRolloutEvent) -> None:
        group = self._groups[rollout.prompt]
        group.append(rollout)
        if len(group) >= self.group_size:
            self._complete.extend(group)
            del self._groups[rollout.prompt]

    def has_batch(self) -> bool:
        return len(self._complete) >= self.batch_size

    def get_batch(self) -> list[TrainingRolloutEvent]:
        batch = self._complete[: self.batch_size]
        self._complete = self._complete[self.batch_size :]
        return batch


class NATSTrainingBridge:
    def __init__(
        self,
        bus: EventBus,
        buffer: RolloutBuffer,
        on_batch_ready: Callable[
            [list[TrainingRolloutEvent]], Coroutine[Any, Any, None]
        ],
        *,
        save_path: Path | None = None,
        trajectory_store: "TrajectoryStore | None" = None,
    ) -> None:
        self.bus = bus
        self.buffer = buffer
        self._on_batch_ready = on_batch_ready
        self._save_path = save_path
        self._trajectory_store = trajectory_store

    async def start(self) -> None:
        await self.bus.subscribe(
            TRAINING_ROLLOUTS, TrainingRolloutEvent, self._handle_rollout
        )
        logger.info("NATSTrainingBridge started")

    async def _handle_rollout(self, rollout: TrainingRolloutEvent) -> None:
        self.buffer.add(rollout)
        if self._save_path:
            self._append_jsonl(rollout)
        if self._trajectory_store is not None:
            self._trajectory_store.add_from_rollout(rollout)
        if self.buffer.has_batch():
            batch = self.buffer.get_batch()
            logger.info("Batch ready: %d rollouts", len(batch))
            await self._on_batch_ready(batch)

    def _append_jsonl(self, rollout: TrainingRolloutEvent) -> None:
        try:
            self._save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._save_path, "a") as f:
                f.write(rollout.model_dump_json() + "\n")
        except OSError as exc:
            logger.warning("Failed to save rollout: %s", exc)
