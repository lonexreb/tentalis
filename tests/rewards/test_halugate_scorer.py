"""Tests for HaluGateScorer — hallucination detection."""

import json
from unittest.mock import AsyncMock

import pytest

from src.rewards.halugate_scorer import (
    DEFAULT_FALLBACK_SCORE,
    DEFAULT_NEUTRAL_SCORE,
    HaluGateScorer,
)


@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def scorer(mock_client):
    return HaluGateScorer(client=mock_client, model="test-model")


@pytest.mark.asyncio
async def test_score_steps_returns_correct_length(scorer, mock_client):
    # No claims → neutral score for all steps
    mock_client.chat.return_value = "[]"
    scores = await scorer.score_steps("task", ["step1", "step2", "step3"])
    assert len(scores) == 3


@pytest.mark.asyncio
async def test_no_claims_returns_neutral(scorer, mock_client):
    mock_client.chat.return_value = "[]"
    scores = await scorer.score_steps("task", ["Just reasoning here"])
    assert scores == [DEFAULT_NEUTRAL_SCORE]


@pytest.mark.asyncio
async def test_supported_claims_score_high(scorer, mock_client):
    mock_client.chat.side_effect = [
        json.dumps(["Earth orbits the Sun"]),  # extraction
        json.dumps({"verdict": "supported", "confidence": 0.95}),  # verification
    ]
    scores = await scorer.score_steps("Astronomy facts", ["Earth orbits the Sun"])
    assert len(scores) == 1
    assert scores[0] == 1.0


@pytest.mark.asyncio
async def test_contradicted_claims_score_low(scorer, mock_client):
    mock_client.chat.side_effect = [
        json.dumps(["The sky is green"]),  # extraction
        json.dumps({"verdict": "contradicted", "confidence": 0.9}),  # verification
    ]
    scores = await scorer.score_steps("Color facts", ["The sky is green"])
    assert scores[0] == 0.0


@pytest.mark.asyncio
async def test_unsupported_claims_score_mid(scorer, mock_client):
    mock_client.chat.side_effect = [
        json.dumps(["Pluto has 7 moons"]),
        json.dumps({"verdict": "unsupported", "confidence": 0.5}),
    ]
    scores = await scorer.score_steps("Planets", ["Pluto has 7 moons"])
    assert scores[0] == 0.5


@pytest.mark.asyncio
async def test_mixed_claims(scorer, mock_client):
    mock_client.chat.side_effect = [
        json.dumps(["claim1", "claim2"]),  # extraction: 2 claims
        json.dumps({"verdict": "supported", "confidence": 0.9}),  # claim1
        json.dumps({"verdict": "contradicted", "confidence": 0.8}),  # claim2
    ]
    scores = await scorer.score_steps("context", ["Two claims here"])
    assert len(scores) == 1
    assert scores[0] == pytest.approx(0.5)  # (1.0 + 0.0) / 2


@pytest.mark.asyncio
async def test_json_parse_error_extraction_fallback(scorer, mock_client):
    mock_client.chat.return_value = "not valid json"
    scores = await scorer.score_steps("task", ["step"])
    # Extraction fails → no claims → neutral
    assert scores == [DEFAULT_NEUTRAL_SCORE]


@pytest.mark.asyncio
async def test_json_parse_error_verification_fallback(scorer, mock_client):
    mock_client.chat.side_effect = [
        json.dumps(["a claim"]),  # extraction succeeds
        "bad json",  # verification fails
    ]
    scores = await scorer.score_steps("task", ["step"])
    assert scores[0] == DEFAULT_FALLBACK_SCORE


@pytest.mark.asyncio
async def test_implements_step_scorer_protocol(scorer):
    from src.rewards.scorer import StepScorer
    assert isinstance(scorer, StepScorer)
