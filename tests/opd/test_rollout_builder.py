"""Tests for CombinedRolloutBuilder — rollout + OPD hint joining."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.events.topics import TRAINING_COMBINED
from src.events.types import (
    CombinedRolloutEvent,
    OPDHintEvent,
    TrainingRolloutEvent,
)
from src.opd.rollout_builder import CombinedRolloutBuilder


@pytest.fixture
def mock_bus():
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def builder(mock_bus):
    return CombinedRolloutBuilder(mock_bus, join_timeout=0.5)


class TestCombinedRolloutBuilder:
    async def test_rollout_then_hint_joins(self, builder, mock_bus):
        rollout = TrainingRolloutEvent(
            task_id="t1", worker_id="w1", prompt="p1",
            response="r1", steps=["s1"], step_scores=[0.8],
            outcome_score=0.8,
        )
        hint = OPDHintEvent(
            task_id="t1", worker_id="w1",
            hint_text="use DP", teacher_logprobs=[-0.5],
            original_response="r1",
        )

        await builder._handle_rollout(rollout)
        await builder._handle_hint(hint)

        mock_bus.publish.assert_called_once()
        combined = mock_bus.publish.call_args[0][1]
        assert isinstance(combined, CombinedRolloutEvent)
        assert combined.task_id == "t1"
        assert combined.has_opd is True
        assert combined.hint_text == "use DP"
        assert combined.outcome_score == 0.8

    async def test_hint_then_rollout_joins(self, builder, mock_bus):
        hint = OPDHintEvent(
            task_id="t2", worker_id="w1",
            hint_text="fix it", teacher_logprobs=[],
        )
        rollout = TrainingRolloutEvent(
            task_id="t2", worker_id="w1", prompt="p2",
            response="r2", outcome_score=0.6,
        )

        await builder._handle_hint(hint)
        await builder._handle_rollout(rollout)

        combined = mock_bus.publish.call_args[0][1]
        assert combined.has_opd is True
        assert combined.hint_text == "fix it"

    async def test_timeout_publishes_pure_rl(self, builder, mock_bus):
        rollout = TrainingRolloutEvent(
            task_id="t3", worker_id="w1", prompt="p3",
            response="r3", outcome_score=0.9,
        )

        await builder._handle_rollout(rollout)

        # Wait for timeout
        await asyncio.sleep(0.7)

        mock_bus.publish.assert_called_once()
        combined = mock_bus.publish.call_args[0][1]
        assert combined.has_opd is False
        assert combined.hint_text == ""
        assert combined.outcome_score == 0.9

    async def test_multiple_tasks_independent(self, builder, mock_bus):
        r1 = TrainingRolloutEvent(
            task_id="t4", worker_id="w1", prompt="p4",
            response="r4", outcome_score=0.5,
        )
        r2 = TrainingRolloutEvent(
            task_id="t5", worker_id="w1", prompt="p5",
            response="r5", outcome_score=0.7,
        )
        h1 = OPDHintEvent(
            task_id="t4", worker_id="w1", hint_text="hint4",
        )

        await builder._handle_rollout(r1)
        await builder._handle_rollout(r2)
        await builder._handle_hint(h1)

        # t4 should be joined, t5 still pending
        assert mock_bus.publish.call_count == 1
        combined = mock_bus.publish.call_args[0][1]
        assert combined.task_id == "t4"
        assert combined.has_opd is True
