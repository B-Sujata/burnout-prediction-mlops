"""
Tests for the questionnaire mapping logic.
All tests run without any API key or trained model (except the last one).
"""

import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Fix 1: correct import path
# ── Fix 2: correct variable name  QUESTIONNAIRE  (not QUESTIONS)
from src.advisor.questionnaire import QUESTIONNAIRE

# ── Fix 4: added missing  sleep_quality  →  now 20 features
EXPECTED_KEYS = {
    "anxiety_level", "self_esteem", "mental_health_history", "depression",
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns", "social_support", "peer_pressure",
    "extracurricular_activities", "bullying",
}

VALID_RANGES = {
    "anxiety_level":               (0, 21),
    "self_esteem":                 (0, 30),
    "mental_health_history":       (0, 1),
    "depression":                  (0, 27),
    "headache":                    (0, 5),
    "blood_pressure":              (1, 3),
    "sleep_quality":               (0, 5),
    "breathing_problem":           (0, 5),
    "noise_level":                 (0, 5),
    "living_conditions":           (0, 5),
    "safety":                      (0, 5),
    "basic_needs":                 (0, 5),
    "academic_performance":        (0, 5),
    "study_load":                  (0, 5),
    "teacher_student_relationship":(0, 5),
    "future_career_concerns":      (0, 5),
    "social_support":              (0, 3),
    "peer_pressure":               (0, 5),
    "extracurricular_activities":  (0, 5),
    "bullying":                    (0, 5),
}


class TestQuestionnaireStructure:

    # ── Fix 5: updated test name to reflect 20 features
    def test_covers_all_20_features(self):
        # ── Fix 3: q["feature"]  (not q["key"])
        keys = {q["feature"] for q in QUESTIONNAIRE}
        missing = EXPECTED_KEYS - keys
        assert not missing, f"Missing features: {missing}"

    def test_no_duplicate_keys(self):
        # ── Fix 3: q["feature"]
        keys = [q["feature"] for q in QUESTIONNAIRE]
        dupes = {k for k in keys if keys.count(k) > 1}
        assert not dupes, f"Duplicate feature keys: {dupes}"

    def test_required_fields_present(self):
        for q in QUESTIONNAIRE:
            for field in ("feature", "question", "options"):
                assert field in q, \
                    f"Missing '{field}' in question for {q.get('feature', '?')}"

    def test_at_least_two_options_per_question(self):
        for q in QUESTIONNAIRE:
            assert len(q["options"]) >= 2, \
                f"{q['feature']}: only {len(q['options'])} option(s)"

    def test_option_values_in_valid_range(self):
        for q in QUESTIONNAIRE:
            # ── Fix 3: q["feature"]
            key = q["feature"]
            if key not in VALID_RANGES:
                continue
            lo, hi = VALID_RANGES[key]
            for label, value in q["options"]:
                assert lo <= value <= hi, \
                    f"{key}: '{label}' → {value} is outside [{lo}, {hi}]"

    def test_option_labels_not_empty(self):
        for q in QUESTIONNAIRE:
            for label, _ in q["options"]:
                assert label.strip(), \
                    f"{q['feature']}: empty option label found"

    def test_option_values_are_numeric(self):
        for q in QUESTIONNAIRE:
            for _, value in q["options"]:
                assert isinstance(value, (int, float)), \
                    f"{q['feature']}: option value {value!r} is not numeric"

    def test_scales_cover_both_ends(self):
        """Every question's options should span from near the low end
        to near the high end of the feature's valid range."""
        for q in QUESTIONNAIRE:
            # ── Fix 3: q["feature"]
            key = q["feature"]
            if key not in VALID_RANGES:
                continue
            lo, hi = VALID_RANGES[key]
            span = hi - lo
            if span == 0:
                continue
            vals = [v for _, v in q["options"]]
            assert min(vals) <= lo + span * 0.4, \
                f"{key}: lowest option value {min(vals)} doesn't reach low end of scale"
            assert max(vals) >= lo + span * 0.6, \
                f"{key}: highest option value {max(vals)} doesn't reach high end of scale"


class TestQuestionnaireMapping:

    def test_first_options_produce_valid_feature_dict(self):
        # ── Fix 3: q["feature"]
        features = {q["feature"]: q["options"][0][1] for q in QUESTIONNAIRE}
        assert set(features.keys()) == EXPECTED_KEYS
        for key, val in features.items():
            if key in VALID_RANGES:
                lo, hi = VALID_RANGES[key]
                assert lo <= val <= hi, \
                    f"{key}: first option value {val} out of [{lo}, {hi}]"

    def test_last_options_produce_valid_feature_dict(self):
        # ── Fix 3: q["feature"]
        features = {q["feature"]: q["options"][-1][1] for q in QUESTIONNAIRE}
        for key, val in features.items():
            if key in VALID_RANGES:
                lo, hi = VALID_RANGES[key]
                assert lo <= val <= hi, \
                    f"{key}: last option value {val} out of [{lo}, {hi}]"

    def test_middle_options_produce_valid_feature_dict(self):
        # ── Fix 3: q["feature"]
        features = {
            q["feature"]: q["options"][len(q["options"]) // 2][1]
            for q in QUESTIONNAIRE
        }
        for key, val in features.items():
            if key in VALID_RANGES:
                lo, hi = VALID_RANGES[key]
                assert lo <= val <= hi, \
                    f"{key}: middle option value {val} out of [{lo}, {hi}]"

    def test_end_to_end_predict_with_middle_answers(self):
        """Middle answers → predict_burnout → valid result with all composite indices."""
        if not os.path.exists("models/burnout_model.pkl"):
            pytest.skip("Run pipeline first to generate trained model")

        import src.utils.config_loader as cl
        cl.ConfigLoader._instance = None
        from src.models.predict_model import predict_burnout

        # ── Fix 3: q["feature"]
        features = {
            q["feature"]: q["options"][len(q["options"]) // 2][1]
            for q in QUESTIONNAIRE
        }
        result = predict_burnout(features)

        assert 0 <= result["burnout_score"] <= 100, \
            f"Score {result['burnout_score']} out of [0, 100]"
        assert result["risk_level"] in ("Low", "Moderate", "High"), \
            f"Invalid risk level: {result['risk_level']}"
        for idx in ("emotional_strain", "physical_stress", "academic_pressure",
                    "social_stress", "recovery_index"):
            assert idx in result, \
                f"Composite index missing from result: {idx}"
            assert 0 <= result[idx] <= 1, \
                f"{idx} = {result[idx]} is out of [0, 1]"