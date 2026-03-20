"""TrainingScheduler — gates training to configured time windows."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timezone

from src.events.types import TrainingRolloutEvent
from src.training.bridge import RolloutBuffer
from src.training.loop import TrainingLoop

logger = logging.getLogger(__name__)


def _parse_schedule(schedule_str: str) -> tuple[time, time]:
    """Parse 'HH:MM-HH:MM' into (start_time, end_time)."""
    start_str, end_str = schedule_str.strip().split("-")
    start_h, start_m = map(int, start_str.strip().split(":"))
    end_h, end_m = map(int, end_str.strip().split(":"))
    return time(start_h, start_m), time(end_h, end_m)


def _in_window(now: time, start: time, end: time) -> bool:
    """Check if the current time falls within the training window."""
    if start <= end:
        return start <= now <= end
    # Overnight window (e.g., 22:00-06:00)
    return now >= start or now <= end


class TrainingScheduler:
    """Wraps TrainingLoop and buffers batches, flushing only during configured windows.

    When the schedule is not active, rollouts accumulate in the buffer.
    A background task periodically checks if the window has opened and
    flushes buffered batches.
    """

    def __init__(
        self,
        training_loop: TrainingLoop,
        *,
        schedule_hours: str = "02:00-06:00",
        check_interval: float = 60.0,
    ) -> None:
        self._loop = training_loop
        self._start_time, self._end_time = _parse_schedule(schedule_hours)
        self._check_interval = check_interval
        self._pending: list[TrainingRolloutEvent] = []
        self._task: asyncio.Task | None = None

    @property
    def is_in_window(self) -> bool:
        now = datetime.now(timezone.utc).time()
        return _in_window(now, self._start_time, self._end_time)

    async def start(self) -> None:
        """Start the scheduler background loop."""
        await self._loop.start()
        self._task = asyncio.create_task(self._schedule_loop())
        logger.info(
            "TrainingScheduler started (window=%s-%s UTC)",
            self._start_time.strftime("%H:%M"),
            self._end_time.strftime("%H:%M"),
        )

    async def _schedule_loop(self) -> None:
        while True:
            await asyncio.sleep(self._check_interval)
            if self.is_in_window and self._pending:
                logger.info(
                    "Training window open — flushing %d pending rollouts",
                    len(self._pending),
                )
                batch = self._pending[:]
                self._pending.clear()
                await self._loop._on_batch(batch)

    def buffer_rollout(self, rollout: TrainingRolloutEvent) -> None:
        """Add a rollout to the pending buffer (used when outside the training window)."""
        self._pending.append(rollout)
        logger.debug(
            "Buffered rollout (pending=%d, in_window=%s)",
            len(self._pending),
            self.is_in_window,
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)
