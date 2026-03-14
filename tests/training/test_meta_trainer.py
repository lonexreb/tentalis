"""Tests for ManagerMetaTrainer — feedback→improvement tracking."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.events.topics import META_ROLLOUTS
from src.events.types import (
    FeedbackEvent,
    ManagerMetaRollout,
    TrainingRolloutEvent,
)
from src.training.meta_trainer import ManagerMetaTrainer


@pytest.fixture
def mock_bus():
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def trainer(mock_bus):
    return ManagerMetaTrainer(
        mock_bus, window_size=50, min_feedback_count=4, batch_size=4,
    )


class TestManagerMetaTrainer:
    async def test_start_subscribes(self, trainer, mock_bus):
        await trainer.start()
        assert mock_bus.subscribe.call_count == 2

    async def test_tracks_rollout_scores(self, trainer):
        for i in range(5):
            rollout = TrainingRolloutEvent(
                task_id=f"t{i}", worker_id="w1", prompt="p",
                response="r", outcome_score=0.5 + i * 0.1,
            )
            await trainer._handle_rollout(rollout)

        mean = trainer._mean_recent_scores("w1")
        assert 0.6 < mean < 0.8

    async def test_ignores_empty_textual_feedback(self, trainer, mock_bus):
        feedback = FeedbackEvent(
            task_id="t1", manager_id="m1", worker_id="w1",
            score=0.5, textual_feedback="",
        )
        await trainer._handle_feedback(feedback)
        mock_bus.publish.assert_not_called()

    async def test_publishes_meta_rollout_after_threshold(self, trainer, mock_bus):
        # Add some rollout scores first
        for i in range(10):
            rollout = TrainingRolloutEvent(
                task_id=f"r{i}", worker_id="w1", prompt="p",
                response="r", outcome_score=0.6,
            )
            await trainer._handle_rollout(rollout)

        # Submit enough feedback to trigger meta-rollout
        for i in range(4):
            feedback = FeedbackEvent(
                task_id=f"f{i}", manager_id="m1", worker_id="w1",
                score=0.5, textual_feedback=f"improve step {i}",
            )
            await trainer._handle_feedback(feedback)

        # Should have published a ManagerMetaRollout
        mock_bus.publish.assert_called_once()
        call_args = mock_bus.publish.call_args
        assert call_args[0][0] == META_ROLLOUTS
        meta = call_args[0][1]
        assert isinstance(meta, ManagerMetaRollout)
        assert meta.manager_id == "m1"
        assert len(meta.feedback_task_ids) == 4

    async def test_improvement_delta_calculation(self, trainer, mock_bus):
        # Phase 1: worker scores low
        for i in range(5):
            rollout = TrainingRolloutEvent(
                task_id=f"r{i}", worker_id="w1", prompt="p",
                response="r", outcome_score=0.3,
            )
            await trainer._handle_rollout(rollout)

        # Manager gives feedback (scores at 0.3 mean)
        for i in range(4):
            feedback = FeedbackEvent(
                task_id=f"f{i}", manager_id="m1", worker_id="w1",
                score=0.4, textual_feedback=f"do better {i}",
            )
            await trainer._handle_feedback(feedback)

        # At this point, worker mean is still 0.3, so delta ≈ 0
        meta = mock_bus.publish.call_args[0][1]
        assert abs(meta.improvement_delta) < 0.01

    async def test_unknown_worker_returns_default_score(self, trainer):
        assert trainer._mean_recent_scores("unknown") == 0.5
