"""Tests for TrainingScheduler — time-window gated training."""

from datetime import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.events.types import TrainingRolloutEvent
from src.training.scheduler import TrainingScheduler, _in_window, _parse_schedule


def _make_rollout() -> TrainingRolloutEvent:
    return TrainingRolloutEvent(
        task_id="t1",
        worker_id="w1",
        prompt="test",
        response="answer",
        steps=["step"],
        step_scores=[0.8],
        outcome_score=0.8,
    )


class TestParseSchedule:
    def test_normal_window(self):
        start, end = _parse_schedule("02:00-06:00")
        assert start == time(2, 0)
        assert end == time(6, 0)

    def test_overnight_window(self):
        start, end = _parse_schedule("22:00-04:00")
        assert start == time(22, 0)
        assert end == time(4, 0)

    def test_with_spaces(self):
        start, end = _parse_schedule(" 08:30 - 12:45 ")
        assert start == time(8, 30)
        assert end == time(12, 45)


class TestInWindow:
    def test_inside_normal_window(self):
        assert _in_window(time(3, 0), time(2, 0), time(6, 0)) is True

    def test_outside_normal_window(self):
        assert _in_window(time(12, 0), time(2, 0), time(6, 0)) is False

    def test_at_start_boundary(self):
        assert _in_window(time(2, 0), time(2, 0), time(6, 0)) is True

    def test_at_end_boundary(self):
        assert _in_window(time(6, 0), time(2, 0), time(6, 0)) is True

    def test_overnight_inside_late(self):
        assert _in_window(time(23, 0), time(22, 0), time(4, 0)) is True

    def test_overnight_inside_early(self):
        assert _in_window(time(3, 0), time(22, 0), time(4, 0)) is True

    def test_overnight_outside(self):
        assert _in_window(time(12, 0), time(22, 0), time(4, 0)) is False


class TestTrainingScheduler:
    @pytest.fixture
    def mock_loop(self):
        loop = MagicMock()
        loop.start = AsyncMock()
        loop._on_batch = AsyncMock()
        return loop

    @pytest.fixture
    def scheduler(self, mock_loop):
        return TrainingScheduler(
            mock_loop,
            schedule_hours="02:00-06:00",
            check_interval=0.1,
        )

    def test_buffer_rollout(self, scheduler):
        rollout = _make_rollout()
        scheduler.buffer_rollout(rollout)
        assert scheduler.pending_count == 1

    def test_buffer_multiple(self, scheduler):
        for _ in range(5):
            scheduler.buffer_rollout(_make_rollout())
        assert scheduler.pending_count == 5

    async def test_start_calls_loop_start(self, scheduler, mock_loop):
        # Patch the schedule loop to not actually run
        with patch.object(scheduler, '_schedule_loop', new_callable=AsyncMock):
            await scheduler.start()
        mock_loop.start.assert_called_once()

    def test_is_in_window(self, scheduler):
        # Just verify the property works without error
        _ = scheduler.is_in_window
