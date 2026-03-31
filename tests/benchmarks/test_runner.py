"""Tests for benchmark runner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.benchmarks.datasets import BenchmarkExample
from src.benchmarks.evaluator import BenchmarkEvaluator, BenchmarkResult


@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.mark.asyncio
async def test_evaluate_empty_dataset(mock_client):
    evaluator = BenchmarkEvaluator(client=mock_client, model="test")
    result = await evaluator.evaluate_dataset([], dataset_name="test")
    assert result.total == 0
    assert result.accuracy == 0.0


@pytest.mark.asyncio
async def test_evaluate_with_correct_answers(mock_client):
    mock_client.chat.return_value = "The answer is #### 42"
    evaluator = BenchmarkEvaluator(client=mock_client, model="test")

    examples = [
        BenchmarkExample(
            dataset="gsm8k", problem_id="0",
            prompt="Q", reference_answer="#### 42",
        ),
    ]
    result = await evaluator.evaluate_dataset(examples, dataset_name="gsm8k")
    assert result.correct == 1
    assert result.accuracy == 1.0


@pytest.mark.asyncio
async def test_evaluate_with_wrong_answers(mock_client):
    mock_client.chat.return_value = "#### 99"
    evaluator = BenchmarkEvaluator(client=mock_client, model="test")

    examples = [
        BenchmarkExample(
            dataset="gsm8k", problem_id="0",
            prompt="Q", reference_answer="#### 42",
        ),
    ]
    result = await evaluator.evaluate_dataset(examples, dataset_name="gsm8k")
    assert result.correct == 0
    assert result.accuracy == 0.0


def test_runner_writes_results(tmp_path):
    """Verify BenchmarkRunner._write_results creates JSON."""
    from src.benchmarks.runner import BenchmarkRunner

    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner._results_dir = tmp_path

    results = {
        "gsm8k": BenchmarkResult(
            dataset="gsm8k", total=10, correct=8, accuracy=0.8,
        ),
    }
    path = runner._write_results(results)
    assert path.exists()

    data = json.loads(path.read_text())
    assert "gsm8k" in data
    assert data["gsm8k"]["accuracy"] == 0.8
