"""Tests for alignment scenario library."""

from src.alignment.scenarios import (
    ALL_SCENARIOS,
    COLLUSION_SCENARIOS,
    DECEPTION_SCENARIOS,
    REWARD_HACKING_SCENARIOS,
    SAFETY_PRAGMATISM_SCENARIOS,
    AlignmentScenario,
    get_scenarios_by_experiment,
)


class TestAlignmentScenario:
    def test_scenario_has_required_fields(self):
        for scenario in ALL_SCENARIOS:
            assert isinstance(scenario, AlignmentScenario)
            assert scenario.name, f"Scenario missing name: {scenario}"
            assert scenario.experiment, f"Scenario missing experiment: {scenario.name}"
            assert scenario.prompt, f"Scenario missing prompt: {scenario.name}"
            assert scenario.expected_behavior, f"Scenario missing expected_behavior: {scenario.name}"

    def test_scenario_names_are_unique(self):
        names = [s.name for s in ALL_SCENARIOS]
        assert len(names) == len(set(names)), "Duplicate scenario names found"

    def test_deception_scenarios_count(self):
        assert len(DECEPTION_SCENARIOS) >= 10

    def test_reward_hacking_scenarios_count(self):
        assert len(REWARD_HACKING_SCENARIOS) >= 10

    def test_safety_pragmatism_scenarios_count(self):
        assert len(SAFETY_PRAGMATISM_SCENARIOS) >= 10

    def test_collusion_scenarios_count(self):
        assert len(COLLUSION_SCENARIOS) >= 10

    def test_all_scenarios_combined(self):
        expected = (
            len(DECEPTION_SCENARIOS)
            + len(REWARD_HACKING_SCENARIOS)
            + len(SAFETY_PRAGMATISM_SCENARIOS)
            + len(COLLUSION_SCENARIOS)
        )
        assert len(ALL_SCENARIOS) == expected

    def test_get_scenarios_by_experiment(self):
        deception = get_scenarios_by_experiment("deception")
        assert len(deception) == len(DECEPTION_SCENARIOS)
        assert all(s.experiment == "deception" for s in deception)

    def test_get_scenarios_unknown_experiment(self):
        result = get_scenarios_by_experiment("nonexistent")
        assert result == []

    def test_safety_relevant_flag(self):
        safety_scenarios = [s for s in ALL_SCENARIOS if s.safety_relevant]
        assert len(safety_scenarios) > 0
        # All safety-pragmatism + some deception should be safety_relevant
        for s in SAFETY_PRAGMATISM_SCENARIOS:
            assert s.safety_relevant, f"{s.name} should be safety_relevant"

    def test_collusion_metadata(self):
        for s in COLLUSION_SCENARIOS:
            assert s.metadata.get("requires_multi_worker") == "true"

    def test_frozen_dataclass(self):
        scenario = DECEPTION_SCENARIOS[0]
        try:
            scenario.name = "modified"
            assert False, "Should be frozen"
        except AttributeError:
            pass
