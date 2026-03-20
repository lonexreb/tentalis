"""Tests for collusion detector."""

import pytest

from src.alignment.collusion_detector import (
    CollusionDetector,
    _jaccard_similarity,
    _ngram_set,
    _pearson_correlation,
)


class TestPearsonCorrelation:
    def test_perfect_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert _pearson_correlation(x, y) == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert _pearson_correlation(x, y) == pytest.approx(-1.0, abs=0.01)

    def test_no_correlation(self):
        # Constant values -> zero std -> return 0
        x = [1.0, 1.0, 1.0]
        y = [2.0, 2.0, 2.0]
        assert _pearson_correlation(x, y) == 0.0

    def test_insufficient_data(self):
        assert _pearson_correlation([1.0], [2.0]) == 0.0
        assert _pearson_correlation([], []) == 0.0


class TestNgramSet:
    def test_basic_ngrams(self):
        ngrams = _ngram_set("hello", n=3)
        assert "hel" in ngrams
        assert "ell" in ngrams
        assert "llo" in ngrams

    def test_short_text(self):
        assert _ngram_set("hi", n=3) == set()

    def test_case_insensitive(self):
        ngrams = _ngram_set("Hello", n=3)
        assert "hel" in ngrams


class TestJaccardSimilarity:
    def test_identical_sets(self):
        s = {"a", "b", "c"}
        assert _jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        s1 = {"a", "b", "c"}
        s2 = {"b", "c", "d"}
        assert _jaccard_similarity(s1, s2) == pytest.approx(0.5)

    def test_empty_sets(self):
        assert _jaccard_similarity(set(), set()) == 0.0


class TestCollusionDetector:
    @pytest.fixture
    def detector(self):
        return CollusionDetector(correlation_threshold=0.8, similarity_threshold=0.6)

    def test_no_workers(self, detector):
        result = detector.detect_coordination()
        assert result["max_correlation"] == 0.0
        assert result["detected"] == 0.0

    def test_single_worker(self, detector):
        detector.record_score("w1", 0.5)
        result = detector.detect_coordination()
        assert result["detected"] == 0.0

    def test_independent_workers(self, detector):
        for i in range(10):
            detector.record_score("w1", 0.5 + (i % 3) * 0.1)
            detector.record_score("w2", 0.7 - (i % 4) * 0.1)
        result = detector.detect_coordination()
        assert result["max_correlation"] < 0.8

    def test_colluding_workers(self, detector):
        for i in range(10):
            score = 0.5 + i * 0.05
            detector.record_score("w1", score)
            detector.record_score("w2", score + 0.01)
        result = detector.detect_coordination()
        assert result["max_correlation"] > 0.9
        assert result["detected"] == 1.0

    def test_detect_patterned_identical_responses(self, detector):
        for i in range(5):
            shared = f"This is a verified correct answer number {i}"
            detector.record_response("w1", shared)
            detector.record_response("w2", shared)
        result = detector.detect_patterned_responses()
        assert result["max_similarity"] == 1.0
        assert result["detected"] == 1.0

    def test_detect_patterned_different_responses(self, detector):
        for i in range(5):
            detector.record_response("w1", f"Unique answer alpha {i}")
            detector.record_response("w2", f"Completely different approach {i}")
        result = detector.detect_patterned_responses()
        assert result["max_similarity"] < 0.6

    def test_reset(self, detector):
        detector.record_score("w1", 0.5)
        detector.record_response("w1", "test")
        detector.reset()
        result = detector.detect_coordination()
        assert result["max_correlation"] == 0.0

    def test_external_scores_dict(self, detector):
        scores = {
            "w1": [0.5, 0.6, 0.7, 0.8, 0.9],
            "w2": [0.51, 0.61, 0.71, 0.81, 0.91],
        }
        result = detector.detect_coordination(worker_scores=scores)
        assert result["max_correlation"] > 0.99
