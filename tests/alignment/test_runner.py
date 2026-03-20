"""Tests for experiment runner."""

import json
from pathlib import Path

import pytest

from src.alignment.runner import ExperimentRunner


@pytest.fixture
def runner(tmp_path):
    return ExperimentRunner(results_dir=str(tmp_path), mock=True)


class TestExperimentRunner:
    async def test_experiment_1_produces_results(self, runner, tmp_path):
        result = await runner.run_experiment_1()
        assert result["experiment"] == "1_alignment_before_after"
        assert "baseline_pass_rate" in result
        assert "post_training_pass_rate" in result
        assert "improvement" in result
        assert result["mock_mode"] is True
        # Check file written
        files = list(tmp_path.glob("exp1_*.json"))
        assert len(files) >= 1

    async def test_experiment_2_detects_hacking(self, runner, tmp_path):
        result = await runner.run_experiment_2()
        assert result["experiment"] == "2_reward_hacking_detection"
        assert "divergence_metrics" in result
        # With misaligned worker, divergence should be detected
        assert result["detection_triggered"] is True

    async def test_experiment_3_safety_pragmatism(self, runner, tmp_path):
        result = await runner.run_experiment_3()
        assert result["experiment"] == "3_safety_pragmatism"
        assert "aligned_pass_rate" in result
        assert "misaligned_pass_rate" in result
        assert "safety_gap" in result

    async def test_experiment_4_collusion(self, runner, tmp_path):
        result = await runner.run_experiment_4()
        assert result["experiment"] == "4_collusion_detection"
        assert "independent_workers" in result
        assert "colluding_workers" in result
        assert result["collusion_detected"] is True

    async def test_experiment_5_audit(self, runner, tmp_path):
        result = await runner.run_experiment_5()
        assert result["experiment"] == "5_audit_trail"
        assert "compliance_mapping" in result
        assert result["compliance_mapping"]["audit_format"] == "JSONL"

    async def test_experiment_6_dashboard(self, runner, tmp_path):
        result = await runner.run_experiment_6()
        assert result["experiment"] == "6_interactive_simulation"
        assert result["dashboard_ready"] is True
        assert result["result_format_valid"] is True

    async def test_run_all(self, runner):
        results = await runner.run_all()
        assert len(results) == 6
        for i in range(1, 7):
            assert f"experiment_{i}" in results

    async def test_results_written_as_json(self, runner, tmp_path):
        await runner.run_experiment_1()
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) >= 1
        for f in json_files:
            data = json.loads(f.read_text())
            assert "experiment" in data or "experiment_name" in data
