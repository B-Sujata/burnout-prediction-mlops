"""
Unit tests for the LLM Burnout Advisor.
These tests validate the advisor logic WITHOUT making real LLM API calls.
"""

import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.advisor.burnout_advisor import identify_stress_drivers, format_stress_drivers


# ── Shared fixture: a sample advisor config ───────────────────────────────────

@pytest.fixture
def advisor_cfg():
    return {
        "inverted_features": [
            "self_esteem", "sleep_quality", "living_conditions",
            "safety", "basic_needs", "academic_performance",
            "teacher_student_relationship", "social_support",
            "extracurricular_activities",
        ],
        "feature_max": {
            "anxiety_level": 21, "self_esteem": 30, "mental_health_history": 1,
            "depression": 27, "headache": 5, "blood_pressure": 3,
            "sleep_quality": 5, "breathing_problem": 5, "noise_level": 5,
            "living_conditions": 5, "safety": 5, "basic_needs": 5,
            "academic_performance": 5, "study_load": 5,
            "teacher_student_relationship": 5, "future_career_concerns": 5,
            "social_support": 3, "peer_pressure": 5,
            "extracurricular_activities": 5, "bullying": 5,
        },
        "feature_min": {"blood_pressure": 1},
        "feature_labels": {
            "anxiety_level": "Anxiety Level",
            "sleep_quality": "Sleep Quality",
            "depression": "Depression",
            "study_load": "Study Load",
            "social_support": "Social Support",
        },
    }


# ── High-stress profile for tests ────────────────────────────────────────────

@pytest.fixture
def high_stress_features():
    return {
        "anxiety_level": 18, "self_esteem": 5, "mental_health_history": 1,
        "depression": 22, "headache": 4, "blood_pressure": 3,
        "sleep_quality": 1, "breathing_problem": 4, "noise_level": 4,
        "living_conditions": 1, "safety": 1, "basic_needs": 1,
        "academic_performance": 1, "study_load": 5, "teacher_student_relationship": 1,
        "future_career_concerns": 5, "social_support": 0, "peer_pressure": 5,
        "extracurricular_activities": 0, "bullying": 5,
    }

@pytest.fixture
def low_stress_features():
    return {
        "anxiety_level": 2, "self_esteem": 28, "mental_health_history": 0,
        "depression": 2, "headache": 0, "blood_pressure": 1,
        "sleep_quality": 5, "breathing_problem": 0, "noise_level": 1,
        "living_conditions": 5, "safety": 5, "basic_needs": 5,
        "academic_performance": 5, "study_load": 1, "teacher_student_relationship": 5,
        "future_career_concerns": 0, "social_support": 3, "peer_pressure": 0,
        "extracurricular_activities": 5, "bullying": 0,
    }


# ── Tests: identify_stress_drivers ───────────────────────────────────────────

class TestStressDriverDetection:

    def test_returns_correct_count(self, high_stress_features, advisor_cfg):
        drivers = identify_stress_drivers(high_stress_features, advisor_cfg, n=3)
        assert len(drivers) == 3

    def test_returns_correct_count_five(self, high_stress_features, advisor_cfg):
        drivers = identify_stress_drivers(high_stress_features, advisor_cfg, n=5)
        assert len(drivers) == 5

    def test_high_stress_top_driver_is_severe(self, high_stress_features, advisor_cfg):
        """Top driver for a high-stress profile should have high contribution."""
        drivers = identify_stress_drivers(high_stress_features, advisor_cfg, n=3)
        assert drivers[0]["stress_contribution"] >= 0.7, \
            f"Expected top driver >= 0.7, got {drivers[0]['stress_contribution']}"

    def test_low_stress_top_driver_is_lower(self, low_stress_features, advisor_cfg):
        """Top driver for a low-stress profile should have low contribution."""
        drivers = identify_stress_drivers(low_stress_features, advisor_cfg, n=3)
        assert drivers[0]["stress_contribution"] <= 0.5, \
            f"Expected top driver <= 0.5, got {drivers[0]['stress_contribution']}"

    def test_drivers_sorted_descending(self, high_stress_features, advisor_cfg):
        drivers = identify_stress_drivers(high_stress_features, advisor_cfg, n=5)
        contributions = [d["stress_contribution"] for d in drivers]
        assert contributions == sorted(contributions, reverse=True)

    def test_driver_dict_has_required_keys(self, high_stress_features, advisor_cfg):
        drivers = identify_stress_drivers(high_stress_features, advisor_cfg, n=1)
        required_keys = {"name", "label", "raw_value", "stress_contribution", "direction"}
        assert required_keys.issubset(set(drivers[0].keys()))

    def test_inverted_feature_direction(self, high_stress_features, advisor_cfg):
        """social_support=0 (inverted) should appear in top drivers with direction='low'."""
        drivers = identify_stress_drivers(high_stress_features, advisor_cfg, n=10)
        low_dir = [d for d in drivers if d["direction"] == "low"]
        assert len(low_dir) > 0, "Expected at least one inverted ('low') driver in top 10"

    def test_contribution_range(self, high_stress_features, advisor_cfg):
        """All contributions should be in [0, 1]."""
        drivers = identify_stress_drivers(high_stress_features, advisor_cfg, n=10)
        for d in drivers:
            assert 0.0 <= d["stress_contribution"] <= 1.0, \
                f"{d['name']}: contribution={d['stress_contribution']} out of [0,1]"


# ── Tests: format_stress_drivers ─────────────────────────────────────────────

class TestFormatStressDrivers:

    def test_output_is_string(self, high_stress_features, advisor_cfg):
        drivers = identify_stress_drivers(high_stress_features, advisor_cfg, n=3)
        result  = format_stress_drivers(drivers)
        assert isinstance(result, str)

    def test_output_has_numbering(self, high_stress_features, advisor_cfg):
        drivers = identify_stress_drivers(high_stress_features, advisor_cfg, n=3)
        result  = format_stress_drivers(drivers)
        assert "1." in result
        assert "2." in result
        assert "3." in result

    def test_output_contains_values(self, high_stress_features, advisor_cfg):
        drivers = identify_stress_drivers(high_stress_features, advisor_cfg, n=1)
        result  = format_stress_drivers(drivers)
        assert str(drivers[0]["raw_value"]) in result

    def test_empty_drivers_returns_empty_string(self):
        assert format_stress_drivers([]) == ""


# ── Tests: predict_burnout with composite indices ─────────────────────────────

class TestPredictWithIndices:

    def test_prediction_includes_composite_indices(self, high_stress_features):
        """predict_burnout should now return composite index values."""
        if not os.path.exists("models/burnout_model.pkl"):
            pytest.skip("Train model first")
        import src.utils.config_loader as cl; cl.ConfigLoader._instance = None
        from src.models.predict_model import predict_burnout
        result = predict_burnout(high_stress_features)
        for key in ["emotional_strain", "physical_stress", "academic_pressure",
                    "social_stress", "recovery_index"]:
            assert key in result, f"Missing composite index: {key}"
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of [0,1]"

    def test_prediction_without_advisor_has_no_llm_advice(self, high_stress_features):
        if not os.path.exists("models/burnout_model.pkl"):
            pytest.skip("Train model first")
        import src.utils.config_loader as cl; cl.ConfigLoader._instance = None
        from src.models.predict_model import predict_burnout
        result = predict_burnout(high_stress_features, use_advisor=False)
        assert "llm_advice" not in result


# ── Tests: advisor config loading ────────────────────────────────────────────

class TestAdvisorConfig:

    def test_advisor_config_exists(self):
        assert os.path.exists("configs/advisor_config.yaml"), \
            "configs/advisor_config.yaml not found"

    def test_advisor_config_has_required_keys(self):
        import yaml
        with open("configs/advisor_config.yaml") as f:
            cfg = yaml.safe_load(f)["advisor"]
        for key in ["provider", "anthropic_model", "openai_model",
                    "temperature", "max_tokens", "persona"]:
            assert key in cfg, f"Missing key in advisor config: {key}"

    def test_provider_is_valid(self):
        import yaml
        with open("configs/advisor_config.yaml") as f:
            cfg = yaml.safe_load(f)["advisor"]
        assert cfg["provider"] in ("openai", "anthropic")

    def test_temperature_in_range(self):
        import yaml
        with open("configs/advisor_config.yaml") as f:
            cfg = yaml.safe_load(f)["advisor"]
        assert 0.0 <= cfg["temperature"] <= 2.0
