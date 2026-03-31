"""Benchmark evaluator — generates responses and checks correctness."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from src.benchmarks.datasets import BenchmarkExample
from src.inference.client import InferenceClient

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from evaluating a benchmark dataset."""

    dataset: str
    total: int
    correct: int
    accuracy: float
    per_difficulty: dict[str, float] = field(default_factory=dict)
    mean_step_score: float = 0.0
    details: list[dict] = field(default_factory=list)


class BenchmarkEvaluator:
    """Runs benchmarks: generate response → extract answer → check correctness."""

    def __init__(
        self,
        client: InferenceClient,
        model: str = "qwen2.5:1.5b",
        scorer: object | None = None,
        max_concurrent: int = 4,
    ) -> None:
        self._client = client
        self._model = model
        self._scorer = scorer
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_dataset(
        self,
        examples: list[BenchmarkExample],
        *,
        dataset_name: str = "unknown",
    ) -> BenchmarkResult:
        if not examples:
            return BenchmarkResult(
                dataset=dataset_name, total=0, correct=0, accuracy=0.0,
            )

        tasks = [self._solve_one(ex) for ex in examples]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        correct = 0
        details: list[dict] = []
        difficulty_counts: dict[str, list[bool]] = {}

        for ex, res in zip(examples, results):
            if isinstance(res, Exception):
                detail = {
                    "problem_id": ex.problem_id, "correct": False,
                    "error": str(res),
                }
            else:
                detail = res
            details.append(detail)

            is_correct = detail.get("correct", False)
            if is_correct:
                correct += 1

            if ex.difficulty:
                difficulty_counts.setdefault(ex.difficulty, []).append(is_correct)

        per_difficulty = {
            level: sum(results_list) / len(results_list)
            for level, results_list in difficulty_counts.items()
        }

        return BenchmarkResult(
            dataset=dataset_name,
            total=len(examples),
            correct=correct,
            accuracy=correct / len(examples) if examples else 0.0,
            per_difficulty=per_difficulty,
            details=details,
        )

    async def _solve_one(self, example: BenchmarkExample) -> dict:
        async with self._semaphore:
            try:
                response = await self._client.chat(
                    self._model,
                    [{"role": "user", "content": example.prompt}],
                )
            except Exception as exc:
                return {
                    "problem_id": example.problem_id,
                    "correct": False,
                    "error": str(exc),
                }

        is_correct = check_answer(
            response, example.reference_answer, example.dataset,
        )
        return {
            "problem_id": example.problem_id,
            "correct": is_correct,
            "response": response[:500],
        }


def check_answer(response: str, reference: str, dataset: str) -> bool:
    """Dataset-specific answer checking."""
    if dataset == "gsm8k":
        return _check_gsm8k(response, reference)
    elif dataset == "math":
        return _check_math(response, reference)
    elif dataset == "humaneval":
        return _check_humaneval(response, reference)
    return response.strip() == reference.strip()


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numerical answer from GSM8K format (#### N)."""
    match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    # Fallback: look for "the answer is N"
    match = re.search(r"(?:the answer is|answer:?)\s*([+-]?\d[\d,]*\.?\d*)", text, re.I)
    if match:
        return match.group(1).replace(",", "")
    return None


def extract_math_answer(text: str) -> str | None:
    r"""Extract answer from \boxed{...} in MATH format."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def _check_gsm8k(response: str, reference: str) -> bool:
    pred = extract_gsm8k_answer(response)
    ref = extract_gsm8k_answer(reference)
    if pred is None or ref is None:
        return False
    try:
        return abs(float(pred) - float(ref)) < 1e-6
    except ValueError:
        return pred == ref


def _check_math(response: str, reference: str) -> bool:
    pred = extract_math_answer(response)
    ref = extract_math_answer(reference)
    if pred is None or ref is None:
        return False
    # Normalize whitespace and compare
    return pred.strip().lower() == ref.strip().lower()


def _check_humaneval(response: str, reference: str) -> bool:
    """Simple string containment check for HumanEval."""
    # Check if the core logic of the reference appears in the response
    ref_lines = [
        l.strip() for l in reference.strip().split("\n") if l.strip()
    ]
    if not ref_lines:
        return False
    # Check if at least the return statement matches
    return any(line in response for line in ref_lines)
