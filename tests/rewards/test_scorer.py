"""Tests for LLMJudgeScorer — mocked InferenceClient, no network required."""

import json
from unittest.mock import AsyncMock

import pytest

from src.rewards.scorer import DEFAULT_FALLBACK_SCORE, LLMJudgeScorer


@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def scorer(mock_client):
    return LLMJudgeScorer(model="test-model", client=mock_client)


def _make_judge_response(progress: float, correctness: float) -> str:
    return json.dumps(
        {"progress": progress, "correctness": correctness, "reasoning": "test"}
    )


async def test_score_steps_calls_per_step(scorer, mock_client):
    mock_client.chat.return_value = _make_judge_response(0.8, 0.6)
    steps = ["step one", "step two", "step three"]
    scores = await scorer.score_steps("solve x+1=2", steps)
    assert len(scores) == 3
    assert mock_client.chat.call_count == 3


async def test_score_computation(scorer, mock_client):
    mock_client.chat.return_value = _make_judge_response(0.8, 0.6)
    scores = await scorer.score_steps("task", ["only step"])
    assert scores == [pytest.approx(0.7)]  # (0.8 + 0.6) / 2


async def test_json_parse_error_fallback(scorer, mock_client):
    mock_client.chat.return_value = "not json at all"
    scores = await scorer.score_steps("task", ["step"])
    assert scores == [DEFAULT_FALLBACK_SCORE]


async def test_missing_keys_fallback(scorer, mock_client):
    mock_client.chat.return_value = "{}"
    scores = await scorer.score_steps("task", ["step"])
    # Both default to 0.5, average = 0.5
    assert scores == [DEFAULT_FALLBACK_SCORE]


# --- Majority voting tests ---


@pytest.fixture
def voting_scorer(mock_client):
    return LLMJudgeScorer(model="test-model", client=mock_client, num_votes=3)


async def test_voting_calls_multiple_times(voting_scorer, mock_client):
    mock_client.chat.return_value = _make_judge_response(0.8, 0.6)
    scores = await voting_scorer.score_steps("task", ["step"])
    # 3 votes per step
    assert mock_client.chat.call_count == 3
    assert len(scores) == 1


async def test_voting_returns_median(voting_scorer, mock_client):
    # Return different scores across the 3 votes
    mock_client.chat.side_effect = [
        _make_judge_response(0.8, 0.8),  # score = 0.8
        _make_judge_response(0.4, 0.4),  # score = 0.4
        _make_judge_response(0.6, 0.6),  # score = 0.6
    ]
    scores = await voting_scorer.score_steps("task", ["step"])
    # Median of [0.4, 0.6, 0.8] = 0.6
    assert scores == [pytest.approx(0.6)]


async def test_voting_handles_partial_failure(mock_client):
    scorer = LLMJudgeScorer(model="test-model", client=mock_client, num_votes=3)
    # One returns bad JSON (falls back to 0.5), two succeed
    mock_client.chat.side_effect = [
        _make_judge_response(0.8, 0.8),  # score = 0.8
        "not json",                       # fallback score = 0.5
        _make_judge_response(0.6, 0.6),  # score = 0.6
    ]
    scores = await scorer.score_steps("task", ["step"])
    # Median of [0.5, 0.6, 0.8] = 0.6
    assert scores == [pytest.approx(0.6)]


async def test_voting_all_fail_returns_fallback(mock_client):
    scorer = LLMJudgeScorer(model="test-model", client=mock_client, num_votes=3)
    mock_client.chat.side_effect = Exception("LLM down")
    scores = await scorer.score_steps("task", ["step"])
    assert scores == [DEFAULT_FALLBACK_SCORE]


async def test_num_votes_one_same_as_default(mock_client):
    scorer = LLMJudgeScorer(model="test-model", client=mock_client, num_votes=1)
    mock_client.chat.return_value = _make_judge_response(0.8, 0.6)
    scores = await scorer.score_steps("task", ["step"])
    # With num_votes=1, should use direct call (not voting)
    assert mock_client.chat.call_count == 1
    assert scores == [pytest.approx(0.7)]
