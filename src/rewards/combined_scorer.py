"""CombinedScorer — composes multiple scorers with per-environment weight profiles."""

from __future__ import annotations

import logging

from src.rewards.scorer import StepScorer

logger = logging.getLogger(__name__)

# Default weight profiles per environment type
DEFAULT_PROFILES: dict[str, dict[str, float]] = {
    "chat": {"prm": 0.6, "halugate": 0.3, "length": 0.1},
    "terminal": {"success": 0.8, "efficiency": 0.2},
    "swe": {"tests": 0.5, "prm": 0.3, "diff_size": 0.2},
    "gui": {"prm": 0.5, "action_validity": 0.3, "efficiency": 0.2},
}


class CombinedScorer:
    """Composes multiple named scorers with environment-specific weight profiles.

    Implements the StepScorer protocol. Scorers not present in the weight
    profile for a given environment are ignored.
    """

    def __init__(
        self,
        scorers: dict[str, StepScorer],
        profiles: dict[str, dict[str, float]] | None = None,
        default_environment: str = "chat",
    ) -> None:
        self._scorers = scorers
        self._profiles = profiles or DEFAULT_PROFILES
        self._default_env = default_environment

    async def score_steps(
        self,
        prompt: str,
        steps: list[str],
        *,
        environment_type: str = "",
    ) -> list[float]:
        env = environment_type or self._default_env
        profile = self._profiles.get(env, self._profiles.get(self._default_env, {}))

        if not profile:
            logger.warning("No weight profile for environment %s", env)
            return [0.5] * len(steps)

        # Collect weighted scores from each available scorer
        weighted_scores: list[list[float]] = []
        weights: list[float] = []

        for scorer_name, weight in profile.items():
            scorer = self._scorers.get(scorer_name)
            if scorer is None:
                continue
            try:
                scores = await scorer.score_steps(prompt, steps)
                weighted_scores.append(scores)
                weights.append(weight)
            except Exception:
                logger.warning(
                    "Scorer %s failed for environment %s", scorer_name, env,
                    exc_info=True,
                )

        if not weighted_scores:
            return [0.5] * len(steps)

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight < 1e-12:
            return [0.5] * len(steps)

        # Combine scores
        combined: list[float] = []
        for step_idx in range(len(steps)):
            score = sum(
                w * scores[step_idx]
                for w, scores in zip(weights, weighted_scores)
            ) / total_weight
            combined.append(score)

        return combined
