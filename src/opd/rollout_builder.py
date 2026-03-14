"""CombinedRolloutBuilder — joins RL rollouts + OPD hints by task_id."""

from __future__ import annotations

import asyncio
import logging

from src.events.bus import EventBus
from src.events.topics import OPD_HINTS, TRAINING_COMBINED, TRAINING_ROLLOUTS
from src.events.types import (
    CombinedRolloutEvent,
    OPDHintEvent,
    TrainingRolloutEvent,
)

logger = logging.getLogger(__name__)


class CombinedRolloutBuilder:
    """Joins TrainingRolloutEvent + OPDHintEvent by task_id.

    If no OPD hint arrives within `join_timeout` seconds after an RL rollout,
    the rollout is published as a pure-RL CombinedRolloutEvent (has_opd=False).
    """

    def __init__(
        self,
        bus: EventBus,
        join_timeout: float = 30.0,
    ) -> None:
        self._bus = bus
        self._join_timeout = join_timeout
        self._pending_rollouts: dict[str, TrainingRolloutEvent] = {}
        self._pending_hints: dict[str, OPDHintEvent] = {}
        self._timers: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        await self._bus.subscribe(
            TRAINING_ROLLOUTS, TrainingRolloutEvent, self._handle_rollout
        )
        await self._bus.subscribe(
            OPD_HINTS, OPDHintEvent, self._handle_hint
        )
        logger.info("CombinedRolloutBuilder started (timeout=%.1fs)", self._join_timeout)

    async def _handle_rollout(self, rollout: TrainingRolloutEvent) -> None:
        tid = rollout.task_id

        # Check if hint already arrived
        hint = self._pending_hints.pop(tid, None)
        if hint:
            await self._publish_combined(rollout, hint)
            return

        # Store and start timeout
        self._pending_rollouts[tid] = rollout
        self._timers[tid] = asyncio.create_task(self._timeout_rollout(tid))

    async def _handle_hint(self, hint: OPDHintEvent) -> None:
        tid = hint.task_id

        # Check if rollout already arrived
        rollout = self._pending_rollouts.pop(tid, None)
        if rollout:
            self._cancel_timer(tid)
            await self._publish_combined(rollout, hint)
            return

        # Store hint, waiting for rollout
        self._pending_hints[tid] = hint

    async def _timeout_rollout(self, task_id: str) -> None:
        await asyncio.sleep(self._join_timeout)
        rollout = self._pending_rollouts.pop(task_id, None)
        if rollout:
            logger.debug("OPD timeout for task %s, publishing pure RL rollout", task_id)
            await self._publish_combined(rollout, hint=None)
        self._timers.pop(task_id, None)

    def _cancel_timer(self, task_id: str) -> None:
        timer = self._timers.pop(task_id, None)
        if timer and not timer.done():
            timer.cancel()

    async def _publish_combined(
        self,
        rollout: TrainingRolloutEvent,
        hint: OPDHintEvent | None,
    ) -> None:
        combined = CombinedRolloutEvent(
            task_id=rollout.task_id,
            worker_id=rollout.worker_id,
            prompt=rollout.prompt,
            response=rollout.response,
            steps=rollout.steps,
            step_scores=rollout.step_scores,
            outcome_score=rollout.outcome_score,
            hint_text=hint.hint_text if hint else "",
            teacher_logprobs=hint.teacher_logprobs if hint else [],
            has_opd=hint is not None,
        )
        await self._bus.publish(TRAINING_COMBINED, combined)
        logger.info(
            "Published CombinedRolloutEvent for task %s (has_opd=%s)",
            rollout.task_id,
            combined.has_opd,
        )
