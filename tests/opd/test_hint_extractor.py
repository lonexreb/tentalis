"""Tests for HintExtractor — mocked InferenceClient + EventBus."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.events.topics import OPD_HINTS
from src.events.types import FeedbackEvent, OPDHintEvent, ResultEvent, TaskStatus
from src.opd.hint_extractor import HintExtractor


@pytest.fixture
def mock_bus():
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.chat = AsyncMock(return_value="Use dynamic programming instead")
    return client


@pytest.fixture
def extractor(mock_bus, mock_client):
    return HintExtractor(mock_bus, mock_client, teacher_model="test-model")


class TestHintExtractor:
    async def test_start_subscribes(self, extractor, mock_bus):
        await extractor.start(["coding"])
        assert mock_bus.subscribe.call_count == 2  # result + feedback

    async def test_skips_empty_feedback(self, extractor, mock_bus):
        feedback = FeedbackEvent(
            task_id="t1",
            manager_id="m1",
            worker_id="w1",
            score=0.8,
            textual_feedback="",
        )
        await extractor._handle_feedback(feedback)
        mock_bus.publish.assert_not_called()

    async def test_extracts_hint_and_publishes(self, extractor, mock_bus, mock_client):
        # Pre-cache a result
        result = ResultEvent(
            task_id="t1",
            worker_id="w1",
            result="def fib(n): return fib(n-1) + fib(n-2)",
            status=TaskStatus.SUCCESS,
            steps=["step1"],
        )
        await extractor._cache_result(result)

        feedback = FeedbackEvent(
            task_id="t1",
            manager_id="m1",
            worker_id="w1",
            score=0.4,
            textual_feedback="Use DP not recursion for fibonacci",
        )
        await extractor._handle_feedback(feedback)

        mock_bus.publish.assert_called_once()
        call_args = mock_bus.publish.call_args
        assert call_args[0][0] == OPD_HINTS
        hint_event = call_args[0][1]
        assert isinstance(hint_event, OPDHintEvent)
        assert hint_event.task_id == "t1"
        assert hint_event.worker_id == "w1"
        assert hint_event.hint_text == "Use dynamic programming instead"
        assert hint_event.original_response == result.result

    async def test_hint_extraction_fallback(self, extractor, mock_bus, mock_client):
        """Falls back to raw feedback text if LLM extraction fails."""
        mock_client.chat.side_effect = [
            RuntimeError("LLM down"),  # hint extraction fails
            "ignored",  # teacher logprob call
        ]

        feedback = FeedbackEvent(
            task_id="t2",
            manager_id="m1",
            worker_id="w1",
            score=0.3,
            textual_feedback="Fix the sorting algorithm",
        )
        await extractor._handle_feedback(feedback)

        hint_event = mock_bus.publish.call_args[0][1]
        assert hint_event.hint_text == "Fix the sorting algorithm"

    async def test_result_cache_lru(self, extractor):
        """Cache evicts oldest entries when full."""
        extractor._cache_size = 2
        for i in range(3):
            result = ResultEvent(
                task_id=f"t{i}",
                worker_id="w1",
                result=f"result-{i}",
                status=TaskStatus.SUCCESS,
            )
            await extractor._cache_result(result)

        assert "t0" not in extractor._result_cache
        assert "t1" in extractor._result_cache
        assert "t2" in extractor._result_cache
