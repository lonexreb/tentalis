"""Tests for hackable scorer and reward hacking detector."""

import pytest

from src.alignment.hackable_scorer import HackableScorer, RewardHackingDetector


class TestHackableScorer:
    @pytest.fixture
    def scorer(self):
        return HackableScorer()

    async def test_high_score_with_keywords(self, scorer):
        steps = ["This is verified correct and accurate"]
        scores = await scorer.score_steps("test prompt", steps)
        assert len(scores) == 1
        assert scores[0] > 0.7  # keywords should boost score

    async def test_low_score_without_keywords(self, scorer):
        steps = ["Simple calculation: 2 + 2 = 4"]
        scores = await scorer.score_steps("test prompt", steps)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(0.3)  # base_score only

    async def test_multiple_steps(self, scorer):
        steps = [
            "Step 1: thinking about this",
            "Step 2: verified correct answer",
            "Step 3: just a normal step",
        ]
        scores = await scorer.score_steps("test prompt", steps)
        assert len(scores) == 3
        assert scores[1] > scores[0]  # step with keyword scores higher
        assert scores[2] == pytest.approx(0.3)

    async def test_score_capped_at_1(self):
        scorer = HackableScorer(keyword_boost=1.0)
        steps = ["verified correct accurate confirmed validated"]
        scores = await scorer.score_steps("test", steps)
        assert scores[0] <= 1.0

    async def test_custom_keywords(self):
        scorer = HackableScorer(keywords=["magic", "special"])
        steps = ["This is a magic answer"]
        scores = await scorer.score_steps("test", steps)
        assert scores[0] > 0.3

    async def test_implements_step_scorer_protocol(self, scorer):
        from src.rewards.scorer import StepScorer
        assert isinstance(scorer, StepScorer)


class TestRewardHackingDetector:
    @pytest.fixture
    def detector(self):
        return RewardHackingDetector(divergence_threshold=0.2)

    def test_no_divergence_initially(self, detector):
        result = detector.detect_divergence(window=5)
        assert result["divergence"] == 0.0
        assert result["detected"] == 0.0

    def test_detects_divergence(self, detector):
        # Hackable scores rising, judge scores falling
        for i in range(10):
            detector.record(0.5 + i * 0.05, 0.5 - i * 0.03)
        result = detector.detect_divergence(window=10)
        assert result["divergence"] > 0.2
        assert result["detected"] == 1.0

    def test_no_divergence_when_aligned(self, detector):
        for i in range(10):
            detector.record(0.7, 0.7)
        result = detector.detect_divergence(window=10)
        assert result["divergence"] == pytest.approx(0.0)
        assert result["detected"] == 0.0

    def test_insufficient_samples(self, detector):
        detector.record(0.9, 0.3)
        result = detector.detect_divergence(window=10)
        assert result["detected"] == 0.0

    def test_reset(self, detector):
        for i in range(10):
            detector.record(0.9, 0.1)
        detector.reset()
        result = detector.detect_divergence(window=10)
        assert result["samples"] == 0.0
