"""Benchmark dataset loading — GSM8K, MATH, HumanEval."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkExample:
    """A single benchmark problem."""

    dataset: str
    problem_id: str
    prompt: str
    reference_answer: str
    difficulty: str = ""


class BenchmarkDataset:
    """Loads benchmark examples from JSONL files."""

    def __init__(self, dataset_dir: str = "benchmark_data") -> None:
        self._dir = Path(dataset_dir)

    def load_gsm8k(
        self, split: str = "test", limit: int | None = None,
    ) -> list[BenchmarkExample]:
        """Load GSM8K examples.

        Expected JSONL format: {"question": str, "answer": str}
        where answer contains "#### <number>" as the final answer.
        """
        path = self._dir / f"gsm8k_{split}.jsonl"
        return self._load_jsonl(
            path, dataset="gsm8k", prompt_key="question",
            answer_key="answer", limit=limit,
        )

    def load_math(
        self, split: str = "test", limit: int | None = None,
    ) -> list[BenchmarkExample]:
        r"""Load MATH examples.

        Expected JSONL format: {"problem": str, "solution": str, "level": str}
        where solution contains \boxed{answer}.
        """
        path = self._dir / f"math_{split}.jsonl"
        return self._load_jsonl(
            path, dataset="math", prompt_key="problem",
            answer_key="solution", difficulty_key="level", limit=limit,
        )

    def load_humaneval(
        self, limit: int | None = None,
    ) -> list[BenchmarkExample]:
        """Load HumanEval examples.

        Expected JSONL format: {"task_id": str, "prompt": str, "canonical_solution": str}
        """
        path = self._dir / "humaneval.jsonl"
        return self._load_jsonl(
            path, dataset="humaneval", prompt_key="prompt",
            answer_key="canonical_solution", id_key="task_id", limit=limit,
        )

    def _load_jsonl(
        self,
        path: Path,
        dataset: str,
        prompt_key: str,
        answer_key: str,
        difficulty_key: str = "",
        id_key: str = "",
        limit: int | None = None,
    ) -> list[BenchmarkExample]:
        if not path.exists():
            logger.warning("Dataset file not found: %s", path)
            return []

        examples: list[BenchmarkExample] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                examples.append(BenchmarkExample(
                    dataset=dataset,
                    problem_id=data.get(id_key, f"{dataset}_{i}") if id_key else f"{dataset}_{i}",
                    prompt=data.get(prompt_key, ""),
                    reference_answer=data.get(answer_key, ""),
                    difficulty=data.get(difficulty_key, "") if difficulty_key else "",
                ))

        logger.info("Loaded %d %s examples from %s", len(examples), dataset, path)
        return examples
