"""Tests for benchmark dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmarks.datasets import BenchmarkDataset, BenchmarkExample


def test_benchmark_example_creation():
    ex = BenchmarkExample(
        dataset="gsm8k",
        problem_id="gsm8k_0",
        prompt="What is 2+2?",
        reference_answer="#### 4",
    )
    assert ex.dataset == "gsm8k"
    assert ex.difficulty == ""


def test_load_gsm8k(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    gsm8k_file = data_dir / "gsm8k_test.jsonl"
    gsm8k_file.write_text(
        json.dumps({"question": "What is 2+2?", "answer": "2+2=4\n#### 4"}) + "\n"
        + json.dumps({"question": "What is 3*3?", "answer": "3*3=9\n#### 9"}) + "\n"
    )

    ds = BenchmarkDataset(str(data_dir))
    examples = ds.load_gsm8k()
    assert len(examples) == 2
    assert examples[0].dataset == "gsm8k"
    assert examples[0].prompt == "What is 2+2?"


def test_load_gsm8k_with_limit(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    gsm8k_file = data_dir / "gsm8k_test.jsonl"
    lines = [json.dumps({"question": f"Q{i}", "answer": f"#### {i}"}) for i in range(10)]
    gsm8k_file.write_text("\n".join(lines) + "\n")

    ds = BenchmarkDataset(str(data_dir))
    examples = ds.load_gsm8k(limit=3)
    assert len(examples) == 3


def test_load_missing_file(tmp_path):
    ds = BenchmarkDataset(str(tmp_path / "nonexistent"))
    examples = ds.load_gsm8k()
    assert examples == []


def test_load_math(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    math_file = data_dir / "math_test.jsonl"
    math_file.write_text(
        json.dumps({
            "problem": "Solve x^2 = 4",
            "solution": "x = \\boxed{2}",
            "level": "Level 1",
        }) + "\n"
    )

    ds = BenchmarkDataset(str(data_dir))
    examples = ds.load_math()
    assert len(examples) == 1
    assert examples[0].difficulty == "Level 1"
