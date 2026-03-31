"""Tests for benchmark evaluator and answer checking."""

from __future__ import annotations

import pytest

from src.benchmarks.evaluator import (
    BenchmarkResult,
    check_answer,
    extract_gsm8k_answer,
    extract_math_answer,
)


def test_extract_gsm8k_answer():
    assert extract_gsm8k_answer("The total is 2+2=4\n#### 4") == "4"
    assert extract_gsm8k_answer("#### 42") == "42"
    assert extract_gsm8k_answer("#### 1,234") == "1234"
    assert extract_gsm8k_answer("no answer here") is None


def test_extract_gsm8k_answer_fallback():
    assert extract_gsm8k_answer("The answer is 42") == "42"
    assert extract_gsm8k_answer("Answer: 7") == "7"


def test_extract_math_answer():
    assert extract_math_answer("Therefore \\boxed{x^2 + 1}") == "x^2 + 1"
    assert extract_math_answer("\\boxed{42}") == "42"
    assert extract_math_answer("no boxed answer") is None


def test_check_answer_gsm8k_correct():
    assert check_answer("#### 42", "#### 42", "gsm8k") is True


def test_check_answer_gsm8k_wrong():
    assert check_answer("#### 43", "#### 42", "gsm8k") is False


def test_check_answer_math_correct():
    assert check_answer("\\boxed{x^2}", "\\boxed{x^2}", "math") is True


def test_check_answer_math_wrong():
    assert check_answer("\\boxed{x^3}", "\\boxed{x^2}", "math") is False


def test_benchmark_result_creation():
    result = BenchmarkResult(
        dataset="gsm8k", total=100, correct=85, accuracy=0.85,
    )
    assert result.accuracy == 0.85
    assert result.per_difficulty == {}
