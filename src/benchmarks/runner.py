"""BenchmarkRunner — orchestrates benchmark execution and result storage."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.benchmarks.datasets import BenchmarkDataset
from src.benchmarks.evaluator import BenchmarkEvaluator, BenchmarkResult
from src.inference.client import InferenceClient

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates full benchmark suite: load → evaluate → write results."""

    def __init__(
        self,
        client: InferenceClient,
        model: str = "qwen2.5:1.5b",
        scorer: object | None = None,
        results_dir: str = "benchmark_results",
        dataset_dir: str = "benchmark_data",
    ) -> None:
        self._client = client
        self._model = model
        self._scorer = scorer
        self._results_dir = Path(results_dir)
        self._datasets = BenchmarkDataset(dataset_dir)
        self._evaluator = BenchmarkEvaluator(
            client=client, model=model, scorer=scorer,
        )

    async def run_all(
        self, *, limit: int | None = None,
    ) -> dict[str, BenchmarkResult]:
        results: dict[str, BenchmarkResult] = {}

        for name, loader in [
            ("gsm8k", lambda: self._datasets.load_gsm8k(limit=limit)),
            ("math", lambda: self._datasets.load_math(limit=limit)),
            ("humaneval", lambda: self._datasets.load_humaneval(limit=limit)),
        ]:
            examples = loader()
            if examples:
                result = await self._evaluator.evaluate_dataset(
                    examples, dataset_name=name,
                )
                results[name] = result
                logger.info(
                    "%s: %d/%d correct (%.1f%%)",
                    name, result.correct, result.total, result.accuracy * 100,
                )
            else:
                logger.warning("No examples loaded for %s", name)

        self._write_results(results)
        return results

    async def run_dataset(
        self,
        dataset: str,
        *,
        limit: int | None = None,
    ) -> BenchmarkResult:
        loaders = {
            "gsm8k": lambda: self._datasets.load_gsm8k(limit=limit),
            "math": lambda: self._datasets.load_math(limit=limit),
            "humaneval": lambda: self._datasets.load_humaneval(limit=limit),
        }
        loader = loaders.get(dataset)
        if loader is None:
            raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(loaders)}")

        examples = loader()
        result = await self._evaluator.evaluate_dataset(
            examples, dataset_name=dataset,
        )
        self._write_results({dataset: result})
        return result

    async def run_comparison(
        self,
        models: list[str],
        *,
        limit: int | None = None,
    ) -> dict[str, dict[str, BenchmarkResult]]:
        all_results: dict[str, dict[str, BenchmarkResult]] = {}

        for model in models:
            self._evaluator = BenchmarkEvaluator(
                client=self._client, model=model, scorer=self._scorer,
            )
            results = await self.run_all(limit=limit)
            all_results[model] = results

        return all_results

    def _write_results(self, results: dict[str, BenchmarkResult]) -> Path:
        self._results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out_path = self._results_dir / f"benchmark_{ts}.json"

        data = {}
        for name, result in results.items():
            data[name] = {
                "dataset": result.dataset,
                "total": result.total,
                "correct": result.correct,
                "accuracy": result.accuracy,
                "per_difficulty": result.per_difficulty,
                "mean_step_score": result.mean_step_score,
            }

        out_path.write_text(json.dumps(data, indent=2))
        logger.info("Results written to %s", out_path)
        return out_path
