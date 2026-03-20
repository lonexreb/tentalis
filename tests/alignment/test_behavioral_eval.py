"""Tests for behavioral evaluation harness."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.alignment.behavioral_eval import (
    AlignmentEvalResult,
    BehavioralEvalHarness,
    LLMJudgeEvaluator,
    PatternBasedEvaluator,
)
from src.alignment.scenarios import AlignmentScenario


@pytest.fixture
def sample_scenario():
    return AlignmentScenario(
        name="test_scenario",
        experiment="test",
        prompt="What is 2+2?",
        expected_behavior="Answer 4",
        misaligned_indicators=["verified correct", "100% certain", "guaranteed"],
    )


@pytest.fixture
def clean_scenario():
    return AlignmentScenario(
        name="clean_scenario",
        experiment="test",
        prompt="What is 2+2?",
        expected_behavior="Answer 4",
        misaligned_indicators=["fabricated", "hallucinated"],
    )


class TestPatternBasedEvaluator:
    @pytest.fixture
    def evaluator(self):
        return PatternBasedEvaluator()

    async def test_detects_misaligned_response(self, evaluator, sample_scenario):
        result = await evaluator.evaluate(
            sample_scenario,
            "The answer is 4. Verified correct.",
            ["Step 1: calculated"],
        )
        assert not result.passed
        assert result.metrics["indicators_matched"] >= 1.0

    async def test_passes_clean_response(self, evaluator, clean_scenario):
        result = await evaluator.evaluate(
            clean_scenario,
            "The answer is 4.",
            ["Step 1: 2 + 2 = 4"],
        )
        assert result.passed
        assert result.metrics["indicators_matched"] == 0.0

    async def test_case_insensitive(self, evaluator, sample_scenario):
        result = await evaluator.evaluate(
            sample_scenario,
            "VERIFIED CORRECT answer",
            [],
        )
        assert not result.passed

    async def test_checks_steps_too(self, evaluator, sample_scenario):
        result = await evaluator.evaluate(
            sample_scenario,
            "Clean response",
            ["Step 1: verified correct"],
        )
        assert not result.passed

    async def test_no_indicators(self, evaluator):
        scenario = AlignmentScenario(
            name="no_indicators",
            experiment="test",
            prompt="test",
            expected_behavior="test",
            misaligned_indicators=[],
        )
        result = await evaluator.evaluate(scenario, "anything", [])
        assert result.passed
        assert result.metrics["misalignment_match_ratio"] == 0.0

    async def test_returns_eval_result(self, evaluator, sample_scenario):
        result = await evaluator.evaluate(sample_scenario, "clean", [])
        assert isinstance(result, AlignmentEvalResult)
        assert result.scenario_name == "test_scenario"


class TestLLMJudgeEvaluator:
    async def test_parses_aligned_response(self):
        mock_client = AsyncMock()
        mock_client.chat.return_value = json.dumps({
            "aligned": True,
            "confidence": 0.9,
            "reasoning": "Response is appropriate",
        })
        evaluator = LLMJudgeEvaluator(client=mock_client)
        scenario = AlignmentScenario(
            name="test", experiment="test", prompt="test",
            expected_behavior="test", misaligned_indicators=["bad"],
        )
        result = await evaluator.evaluate(scenario, "good response", ["step 1"])
        assert result.passed
        assert result.metrics["confidence"] == 0.9

    async def test_parses_misaligned_response(self):
        mock_client = AsyncMock()
        mock_client.chat.return_value = json.dumps({
            "aligned": False,
            "confidence": 0.8,
            "reasoning": "Contains fabrication",
        })
        evaluator = LLMJudgeEvaluator(client=mock_client)
        scenario = AlignmentScenario(
            name="test", experiment="test", prompt="test",
            expected_behavior="test", misaligned_indicators=[],
        )
        result = await evaluator.evaluate(scenario, "bad response", [])
        assert not result.passed
        assert result.metrics["confidence"] == 0.8

    async def test_handles_json_error(self):
        mock_client = AsyncMock()
        mock_client.chat.return_value = "not json"
        evaluator = LLMJudgeEvaluator(client=mock_client)
        scenario = AlignmentScenario(
            name="test", experiment="test", prompt="test",
            expected_behavior="test", misaligned_indicators=[],
        )
        result = await evaluator.evaluate(scenario, "response", [])
        assert result.passed  # defaults to pass on error
        assert result.metrics.get("error") == 1.0


class TestBehavioralEvalHarness:
    async def test_runs_scenarios_and_produces_results(self, tmp_path):
        async def mock_worker(prompt: str) -> tuple[str, list[str]]:
            return "clean answer", ["step 1"]

        evaluator = PatternBasedEvaluator()
        harness = BehavioralEvalHarness(
            evaluator=evaluator,
            worker_fn=mock_worker,
            experiment_name="test_exp",
            results_dir=str(tmp_path),
        )
        scenarios = [
            AlignmentScenario(
                name="s1", experiment="test", prompt="test",
                expected_behavior="test", misaligned_indicators=["bad"],
            ),
            AlignmentScenario(
                name="s2", experiment="test", prompt="test",
                expected_behavior="test", misaligned_indicators=["bad"],
            ),
        ]
        results = await harness.run(scenarios)
        assert len(results) == 2
        assert all(r.passed for r in results)

        # Check JSON file was written
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["total_scenarios"] == 2
        assert data["passed"] == 2

    async def test_handles_worker_error(self, tmp_path):
        async def failing_worker(prompt: str) -> tuple[str, list[str]]:
            raise RuntimeError("Worker crashed")

        evaluator = PatternBasedEvaluator()
        harness = BehavioralEvalHarness(
            evaluator=evaluator,
            worker_fn=failing_worker,
            experiment_name="test_error",
            results_dir=str(tmp_path),
        )
        scenarios = [
            AlignmentScenario(
                name="s1", experiment="test", prompt="test",
                expected_behavior="test", misaligned_indicators=[],
            ),
        ]
        results = await harness.run(scenarios)
        assert len(results) == 1
        assert not results[0].passed
        assert "Worker error" in results[0].details

    async def test_publishes_events_when_bus_provided(self, tmp_path):
        async def mock_worker(prompt: str) -> tuple[str, list[str]]:
            return "answer", ["step"]

        mock_bus = AsyncMock()
        evaluator = PatternBasedEvaluator()
        harness = BehavioralEvalHarness(
            evaluator=evaluator,
            worker_fn=mock_worker,
            experiment_name="test_bus",
            results_dir=str(tmp_path),
            bus=mock_bus,
        )
        scenarios = [
            AlignmentScenario(
                name="s1", experiment="test", prompt="test",
                expected_behavior="test", misaligned_indicators=[],
            ),
        ]
        await harness.run(scenarios, worker_id="w1")
        mock_bus.publish.assert_called_once()
