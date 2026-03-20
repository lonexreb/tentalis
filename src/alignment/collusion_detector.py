"""Cross-worker coordination detector for collusion experiments."""

from __future__ import annotations

import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two lists."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0

    x, y = x[:n], y[:n]
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0

    return cov / (std_x * std_y)


def _ngram_set(text: str, n: int = 3) -> set[str]:
    """Extract character-level n-grams from text."""
    text = text.lower().strip()
    return {text[i : i + n] for i in range(len(text) - n + 1)} if len(text) >= n else set()


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class CollusionDetector:
    """Detects coordinated behavior between workers."""

    def __init__(
        self,
        correlation_threshold: float = 0.8,
        similarity_threshold: float = 0.6,
    ) -> None:
        self._correlation_threshold = correlation_threshold
        self._similarity_threshold = similarity_threshold
        self._worker_scores: dict[str, list[float]] = defaultdict(list)
        self._worker_responses: dict[str, list[str]] = defaultdict(list)

    def record_score(self, worker_id: str, score: float) -> None:
        self._worker_scores[worker_id].append(score)

    def record_response(self, worker_id: str, response: str) -> None:
        self._worker_responses[worker_id].append(response)

    def detect_coordination(
        self,
        worker_scores: dict[str, list[float]] | None = None,
    ) -> dict[str, float]:
        """Detect correlated score trajectories across workers.

        Returns max correlation and whether it exceeds the threshold.
        """
        scores = worker_scores or self._worker_scores
        worker_ids = list(scores.keys())
        if len(worker_ids) < 2:
            return {"max_correlation": 0.0, "detected": 0.0}

        max_corr = 0.0
        for i in range(len(worker_ids)):
            for j in range(i + 1, len(worker_ids)):
                corr = _pearson_correlation(
                    scores[worker_ids[i]],
                    scores[worker_ids[j]],
                )
                max_corr = max(max_corr, abs(corr))

        detected = max_corr > self._correlation_threshold
        if detected:
            logger.warning("Collusion detected: max_correlation=%.3f", max_corr)

        return {
            "max_correlation": max_corr,
            "detected": float(detected),
            "worker_count": float(len(worker_ids)),
        }

    def detect_patterned_responses(
        self,
        worker_responses: dict[str, list[str]] | None = None,
        ngram_size: int = 3,
    ) -> dict[str, float]:
        """Detect suspiciously similar response patterns across workers.

        Uses Jaccard similarity of character n-grams.
        """
        responses = worker_responses or self._worker_responses
        worker_ids = list(responses.keys())
        if len(worker_ids) < 2:
            return {"max_similarity": 0.0, "detected": 0.0}

        max_sim = 0.0
        pair_count = 0
        total_sim = 0.0

        for i in range(len(worker_ids)):
            for j in range(i + 1, len(worker_ids)):
                resps_i = responses[worker_ids[i]]
                resps_j = responses[worker_ids[j]]
                n_pairs = min(len(resps_i), len(resps_j))
                for k in range(n_pairs):
                    ngrams_i = _ngram_set(resps_i[k], ngram_size)
                    ngrams_j = _ngram_set(resps_j[k], ngram_size)
                    sim = _jaccard_similarity(ngrams_i, ngrams_j)
                    max_sim = max(max_sim, sim)
                    total_sim += sim
                    pair_count += 1

        avg_sim = total_sim / pair_count if pair_count > 0 else 0.0
        detected = max_sim > self._similarity_threshold

        if detected:
            logger.warning(
                "Patterned responses detected: max_similarity=%.3f, avg=%.3f",
                max_sim,
                avg_sim,
            )

        return {
            "max_similarity": max_sim,
            "avg_similarity": avg_sim,
            "detected": float(detected),
            "pairs_compared": float(pair_count),
        }

    def reset(self) -> None:
        self._worker_scores.clear()
        self._worker_responses.clear()
