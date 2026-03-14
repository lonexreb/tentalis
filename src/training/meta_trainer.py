"""ManagerMetaTrainer — outer-loop RL that trains the manager to give better feedback."""

from __future__ import annotations

import logging
from collections import defaultdict, deque

from src.events.bus import EventBus
from src.events.topics import FEEDBACK_SCORED, META_ROLLOUTS, TRAINING_ROLLOUTS
from src.events.types import (
    FeedbackEvent,
    ManagerMetaRollout,
    TrainingRolloutEvent,
)

logger = logging.getLogger(__name__)


class ManagerMetaTrainer:
    """Tracks feedback → improvement correlation to train the manager.

    For each worker, maintains a sliding window of (feedback_event, pre_score, post_score).
    When the window has enough samples, computes the improvement delta as the
    manager's meta-reward signal and publishes ManagerMetaRollout.

    The improvement delta is:
        mean(post_scores) - mean(pre_scores) for tasks following the feedback
    """

    def __init__(
        self,
        bus: EventBus,
        window_size: int = 200,
        min_feedback_count: int = 200,
        batch_size: int = 16,
    ) -> None:
        self._bus = bus
        self._window_size = window_size
        self._min_feedback = min_feedback_count
        self._batch_size = batch_size

        # worker_id → deque of (task_id, score) pairs
        self._worker_scores: dict[str, deque[tuple[str, float]]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        # manager_id → list of (task_id, feedback_text, worker_id, score_at_feedback)
        self._feedback_log: dict[str, list[tuple[str, str, str, float]]] = defaultdict(list)
        self._total_feedback = 0

    async def start(self) -> None:
        await self._bus.subscribe(
            TRAINING_ROLLOUTS, TrainingRolloutEvent, self._handle_rollout
        )
        await self._bus.subscribe(
            FEEDBACK_SCORED, FeedbackEvent, self._handle_feedback
        )
        logger.info(
            "ManagerMetaTrainer started (window=%d, min_feedback=%d)",
            self._window_size, self._min_feedback,
        )

    async def _handle_rollout(self, rollout: TrainingRolloutEvent) -> None:
        self._worker_scores[rollout.worker_id].append(
            (rollout.task_id, rollout.outcome_score)
        )

    async def _handle_feedback(self, feedback: FeedbackEvent) -> None:
        if not feedback.textual_feedback:
            return

        # Record score at time of feedback
        worker_scores = self._worker_scores.get(feedback.worker_id)
        current_mean = self._mean_recent_scores(feedback.worker_id)

        self._feedback_log[feedback.manager_id].append(
            (feedback.task_id, feedback.textual_feedback, feedback.worker_id, current_mean)
        )
        self._total_feedback += 1

        # Check if we have enough data for a meta-training step
        if self._total_feedback >= self._min_feedback:
            await self._maybe_publish_meta_rollout(feedback.manager_id)

    async def _maybe_publish_meta_rollout(self, manager_id: str) -> None:
        log = self._feedback_log.get(manager_id, [])
        if len(log) < self._batch_size:
            return

        # Take oldest batch
        batch = log[: self._batch_size]
        self._feedback_log[manager_id] = log[self._batch_size :]

        # Compute improvement delta: current worker performance vs at-feedback
        task_ids = []
        feedback_texts = []
        score_befores = []

        for task_id, text, worker_id, score_at in batch:
            task_ids.append(task_id)
            feedback_texts.append(text)
            score_befores.append(score_at)

        mean_before = sum(score_befores) / len(score_befores) if score_befores else 0.0

        # Get current mean across all workers that received feedback
        worker_ids = {b[2] for b in batch}
        current_scores = [
            self._mean_recent_scores(wid) for wid in worker_ids
        ]
        mean_after = (
            sum(current_scores) / len(current_scores) if current_scores else 0.0
        )

        improvement = mean_after - mean_before

        meta_rollout = ManagerMetaRollout(
            manager_id=manager_id,
            feedback_task_ids=task_ids,
            improvement_delta=improvement,
            feedback_texts=feedback_texts,
            mean_score_before=mean_before,
            mean_score_after=mean_after,
        )
        await self._bus.publish(META_ROLLOUTS, meta_rollout)
        logger.info(
            "Published ManagerMetaRollout: delta=%.4f (before=%.3f, after=%.3f)",
            improvement, mean_before, mean_after,
        )

    def _mean_recent_scores(self, worker_id: str) -> float:
        scores = self._worker_scores.get(worker_id)
        if not scores:
            return 0.5
        vals = [s for _, s in scores]
        return sum(vals) / len(vals)
