"""Tests for CombinedScorer — environment-aware multi-scorer composition."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.rewards.combined_scorer import CombinedScorer


class MockScorer:
    def __init__(self, fixed_scores: list[float]):
        self._scores = fixed_scores

    async def score_steps(self, prompt: str, steps: list[str]) -> list[float]:
        return self._scores[: len(steps)]


class TestCombinedScorer:
    async def test_single_scorer_passthrough(self):
        scorer = CombinedScorer(
            scorers={"prm": MockScorer([0.8, 0.6])},
            profiles={"chat": {"prm": 1.0}},
        )
        scores = await scorer.score_steps("task", ["s1", "s2"], environment_type="chat")
        assert scores == [0.8, 0.6]

    async def test_weighted_combination(self):
        scorer = CombinedScorer(
            scorers={
                "prm": MockScorer([0.8]),
                "halugate": MockScorer([0.4]),
            },
            profiles={"chat": {"prm": 0.6, "halugate": 0.4}},
        )
        scores = await scorer.score_steps("task", ["s1"], environment_type="chat")
        # (0.8 * 0.6 + 0.4 * 0.4) / 1.0 = 0.64
        assert pytest.approx(scores[0], abs=1e-6) == 0.64

    async def test_missing_scorer_ignored(self):
        scorer = CombinedScorer(
            scorers={"prm": MockScorer([0.8])},
            profiles={"chat": {"prm": 0.6, "halugate": 0.4}},
        )
        scores = await scorer.score_steps("task", ["s1"], environment_type="chat")
        # Only prm available, weight normalized to 1.0
        assert scores == [0.8]

    async def test_different_environments(self):
        scorer = CombinedScorer(
            scorers={
                "prm": MockScorer([0.8]),
                "success": MockScorer([1.0]),
            },
            profiles={
                "chat": {"prm": 1.0},
                "terminal": {"success": 1.0},
            },
        )
        chat_scores = await scorer.score_steps("task", ["s1"], environment_type="chat")
        term_scores = await scorer.score_steps("task", ["s1"], environment_type="terminal")
        assert chat_scores == [0.8]
        assert term_scores == [1.0]

    async def test_unknown_environment_falls_back(self):
        scorer = CombinedScorer(
            scorers={"prm": MockScorer([0.5])},
            profiles={"chat": {"prm": 1.0}},
            default_environment="chat",
        )
        scores = await scorer.score_steps("task", ["s1"], environment_type="unknown")
        # Falls back to default_environment's profile
        assert scores == [0.5]

    async def test_no_scorers_returns_default(self):
        scorer = CombinedScorer(
            scorers={},
            profiles={"chat": {"prm": 1.0}},
        )
        scores = await scorer.score_steps("task", ["s1", "s2"], environment_type="chat")
        assert scores == [0.5, 0.5]

    async def test_scorer_failure_handled(self):
        failing_scorer = AsyncMock()
        failing_scorer.score_steps = AsyncMock(side_effect=RuntimeError("boom"))

        scorer = CombinedScorer(
            scorers={
                "prm": MockScorer([0.8]),
                "bad": failing_scorer,
            },
            profiles={"chat": {"prm": 0.5, "bad": 0.5}},
        )
        scores = await scorer.score_steps("task", ["s1"], environment_type="chat")
        # Only prm succeeds, so result is [0.8]
        assert scores == [0.8]
