"""HaluGate — hallucination detection as a StepScorer.

Scores each step by extracting factual claims and verifying them against
the task context via NLI (natural language inference). Steps with only
reasoning (no factual claims) get a neutral score.
"""

from __future__ import annotations

import asyncio
import json
import logging

from src.inference.client import InferenceClient

logger = logging.getLogger(__name__)

DEFAULT_NEUTRAL_SCORE = 0.8
DEFAULT_FALLBACK_SCORE = 0.5

VERDICT_SCORES = {
    "supported": 1.0,
    "contradicted": 0.0,
    "unsupported": 0.5,
}

CLAIM_EXTRACTION_PROMPT = """\
Extract factual claims from this reasoning step. A factual claim is a \
statement that asserts something concrete and verifiable (numbers, names, \
properties, relationships). Pure reasoning or logic steps are NOT claims.

Return ONLY a JSON array of claim strings. If no factual claims, return [].

Step: "{step}"

JSON array:"""

CLAIM_VERIFICATION_PROMPT = """\
Given the following context and claim, classify the claim as:
- "supported": the context contains evidence supporting this claim
- "contradicted": the context contains evidence contradicting this claim
- "unsupported": the context does not contain relevant evidence

Context: {context}

Claim: "{claim}"

Respond with ONLY a JSON object:
{{"verdict": "supported", "confidence": 0.9}}"""


class HaluGateScorer:
    """Hallucination detection scorer implementing StepScorer protocol.

    Pipeline per step:
    1. Extract factual claims via LLM.
    2. Verify each claim against the prompt context via LLM NLI.
    3. Score = mean of verdict scores. No claims → neutral score (0.8).
    """

    def __init__(
        self,
        *,
        client: InferenceClient,
        model: str = "qwen2.5:1.5b",
    ) -> None:
        self._client = client
        self._model = model

    async def score_steps(self, prompt: str, steps: list[str]) -> list[float]:
        scores: list[float] = []
        for step in steps:
            score = await self._score_single_step(prompt, step)
            scores.append(score)
        return scores

    async def _score_single_step(self, context: str, step: str) -> float:
        claims = await self._extract_claims(step)
        if not claims:
            return DEFAULT_NEUTRAL_SCORE

        verdicts = await asyncio.gather(
            *(self._verify_claim(claim, context) for claim in claims)
        )
        return sum(verdicts) / len(verdicts)

    async def _extract_claims(self, step: str) -> list[str]:
        prompt_text = CLAIM_EXTRACTION_PROMPT.format(step=step)
        try:
            response = await self._client.chat(
                self._model,
                [{"role": "user", "content": prompt_text}],
            )
            claims = json.loads(response)
            if isinstance(claims, list):
                return [str(c) for c in claims]
            return []
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Claim extraction failed: %s", exc)
            return []

    async def _verify_claim(self, claim: str, context: str) -> float:
        prompt_text = CLAIM_VERIFICATION_PROMPT.format(
            context=context, claim=claim,
        )
        try:
            response = await self._client.chat(
                self._model,
                [{"role": "user", "content": prompt_text}],
            )
            data = json.loads(response)
            verdict = data.get("verdict", "unsupported")
            return VERDICT_SCORES.get(verdict, DEFAULT_FALLBACK_SCORE)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Claim verification failed: %s", exc)
            return DEFAULT_FALLBACK_SCORE
