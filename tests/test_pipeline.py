"""
Unit tests for the Burnout Prediction pipeline (Real Dataset Edition).
Run with: pytest tests/ -v
"""

import os, sys, pytest, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config():
    from src.utils.config_loader import load_config
    import src.utils.config_loader as cl; cl.ConfigLoader._instance = None
    return load_config("configs/config.yaml")

@pytest.fixture(scope="module")
def sample_df():
    """Minimal dataframe with real dataset column names and realistic ranges."""
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame({
        "anxiety_level":                rng.integers(0, 22, n),
        "self_esteem":                  rng.integers(0, 31, n),
        "mental_health_history":        rng.integers(0, 2,  n),
        "depression":                   rng.integers(0, 28, n),
        "headache":                     rng.integers(0, 6,  n),
        "blood_pressure":               rng.integers(1, 4,  n),
        "sleep_quality":                rng.integers(0, 6,  n),
        "breathing_problem":            rng.integers(0, 6,  n),
        "noise_level":                  rng.integers(0, 6,  n),
        "living_conditions":            rng.integers(0, 6,  n),
        "safety":                       rng.integers(0, 6,  n),
        "basic_needs":                  rng.integers(0, 6,  n),
        "academic_performance":         rng.integers(0, 6,  n),
        "study_load":                   rng.integers(0, 6,  n),
        "teacher_student_relationship": rng.integers(0, 6,  n),
        "future_career_concerns":       rng.integers(0, 6,  n),
        "social_support":               rng.integers(0, 4,  n),
        "peer_pressure":                rng.integers(0, 6,  n),
        "extracurricular_activities":   rng.integers(0, 6,  n),
        "bullying":                     rng.integers(0, 6,  n),
        "stress_level":                 rng.integers(0, 3,  n),
    })


# ── Ingestion ─────────────────────────────────────────────────────────────────

class TestIngestion:
    def test_real_csv_exists(self):
        assert os.path.exists("data/raw/student_data.csv"), \
            "Place StressLevelDataset.csv at data/raw/student_data.csv"

    def test_real_csv_shape(self):
        df = pd.read_csv("data/raw/student_data.csv")
        assert df.shape == (1100, 21), f"Expected (1100, 21), got {df.shape}"

    def test_real_csv_no_missing(self):
        df = pd.read_csv("data/raw/student_data.csv")
        assert df.isnull().sum().sum() == 0

    def test_real_csv_no_duplicates(self):
        df = pd.read_csv("data/raw/student_data.csv")
        assert df.duplicated().sum() == 0

    def test_stress_level_balanced(self):
        df = pd.read_csv("data/raw/student_data.csv")
        counts = df["stress_level"].value_counts()
        # Each class should have between 300 and 500 samples
        for lvl in [0, 1, 2]:
            assert 250 <= counts[lvl] <= 550, f"stress_level={lvl} count={counts[lvl]}"


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_standardize_column_names(self):
        from src.data.preprocess_data import standardize_column_names
        df = pd.DataFrame({"  Anxiety Level ": [1], "Study Load": [2]})
        df = standardize_column_names(df)
        assert "anxiety_level" in df.columns
        assert "study_load" in df.columns

    def test_handle_missing_values(self, sample_df):
        from src.data.preprocess_data import handle_missing_values
        df = sample_df.copy().astype(float)
        df.loc[:5, "anxiety_level"] = np.nan
        df = handle_missing_values(df)
        assert df["anxiety_level"].isnull().sum() == 0

    def test_no_duplicates_after_removal(self, sample_df):
        from src.data.preprocess_data import remove_duplicates
        df = pd.concat([sample_df, sample_df.iloc[:10]], ignore_index=True)
        df = remove_duplicates(df)
        assert df.duplicated().sum() == 0


# ── Feature Engineering ───────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_emotional_strain_range(self, sample_df, config):
        from src.data.feature_engineering import build_emotional_strain
        fw = config.get("feature_weights", "emotional_strain")
        s = build_emotional_strain(sample_df, fw)
        assert s.min() >= 0, "emotional_strain < 0"
        assert s.max() <= 1.01, "emotional_strain > 1"
        assert s.isnull().sum() == 0

    def test_physical_stress_range(self, sample_df, config):
        from src.data.feature_engineering import build_physical_stress
        fw = config.get("feature_weights", "physical_stress")
        s = build_physical_stress(sample_df, fw)
        assert s.min() >= 0
        assert s.max() <= 1.01
        assert s.isnull().sum() == 0

    def test_recovery_index_range(self, sample_df, config):
        from src.data.feature_engineering import build_recovery_index
        fw = config.get("feature_weights", "recovery_index")
        s = build_recovery_index(sample_df, fw)
        assert s.min() >= 0
        assert s.max() <= 1.01

    def test_burnout_score_range(self, sample_df, config):
        from src.data.feature_engineering import (
            build_emotional_strain, build_physical_stress,
            build_academic_pressure, build_social_stress,
            build_recovery_index, compute_burnout_score
        )
        fw = config.get("feature_weights")
        formula = config.get("burnout_formula")
        df = sample_df.copy()
        df["emotional_strain"]  = build_emotional_strain(df,  fw["emotional_strain"])
        df["physical_stress"]   = build_physical_stress(df,   fw["physical_stress"])
        df["academic_pressure"] = build_academic_pressure(df, fw["academic_pressure"])
        df["social_stress"]     = build_social_stress(df,     fw["social_stress"])
        df["recovery_index"]    = build_recovery_index(df,    fw["recovery_index"])
        score = compute_burnout_score(df, formula)
        assert score.min() >= 0
        assert score.max() <= 100
        assert score.isnull().sum() == 0

    def test_burnout_aligns_with_stress_level(self):
        """Mean burnout score should increase with stress_level 0→1→2."""
        if not os.path.exists("data/processed/engineered_data.csv"):
            pytest.skip("Run pipeline first to generate engineered data")
        df = pd.read_csv("data/processed/engineered_data.csv")
        means = df.groupby("stress_level")["burnout_score"].mean().sort_index()
        assert means[0] < means[1] < means[2], \
            f"Burnout scores not monotonically increasing: {means.to_dict()}"


# ── Prediction ────────────────────────────────────────────────────────────────

class TestPrediction:
    def test_classify_low(self):
        from src.models.predict_model import classify_risk
        assert classify_risk(20)["risk_level"] == "Low"

    def test_classify_moderate(self):
        from src.models.predict_model import classify_risk
        assert classify_risk(50)["risk_level"] == "Moderate"

    def test_classify_high(self):
        from src.models.predict_model import classify_risk
        assert classify_risk(80)["risk_level"] == "High"

    def test_boundary_33_low(self):
        from src.models.predict_model import classify_risk
        assert classify_risk(33)["risk_level"] == "Low"

    def test_boundary_34_moderate(self):
        from src.models.predict_model import classify_risk
        assert classify_risk(34)["risk_level"] == "Moderate"

    def test_boundary_66_moderate(self):
        from src.models.predict_model import classify_risk
        assert classify_risk(66)["risk_level"] == "Moderate"

    def test_boundary_67_high(self):
        from src.models.predict_model import classify_risk
        assert classify_risk(67)["risk_level"] == "High"

    def test_predict_high_profile(self):
        if not os.path.exists("models/burnout_model.pkl"):
            pytest.skip("Train model first")
        from src.models.predict_model import predict_burnout
        import src.utils.config_loader as cl; cl.ConfigLoader._instance = None
        high = {
            'anxiety_level':18,'self_esteem':5,'mental_health_history':1,'depression':22,
            'headache':4,'blood_pressure':3,'sleep_quality':1,'breathing_problem':4,
            'noise_level':4,'living_conditions':1,'safety':1,'basic_needs':1,
            'academic_performance':1,'study_load':5,'teacher_student_relationship':1,
            'future_career_concerns':5,'social_support':0,'peer_pressure':5,
            'extracurricular_activities':0,'bullying':5
        }
        r = predict_burnout(high)
        assert r["burnout_score"] >= 60
        assert r["risk_level"] == "High"

    def test_predict_low_profile(self):
        if not os.path.exists("models/burnout_model.pkl"):
            pytest.skip("Train model first")
        from src.models.predict_model import predict_burnout
        import src.utils.config_loader as cl; cl.ConfigLoader._instance = None
        low = {
            'anxiety_level':2,'self_esteem':28,'mental_health_history':0,'depression':2,
            'headache':0,'blood_pressure':1,'sleep_quality':5,'breathing_problem':0,
            'noise_level':1,'living_conditions':5,'safety':5,'basic_needs':5,
            'academic_performance':5,'study_load':1,'teacher_student_relationship':5,
            'future_career_concerns':0,'social_support':3,'peer_pressure':0,
            'extracurricular_activities':5,'bullying':0
        }
        r = predict_burnout(low)
        assert r["burnout_score"] <= 40
        assert r["risk_level"] == "Low"


# ── Config Loader ─────────────────────────────────────────────────────────────

class TestConfigLoader:
    def test_load_succeeds(self, config):
        assert config is not None

    def test_nested_value(self, config):
        seed = config.get("project", "random_seed")
        assert isinstance(seed, int) and seed > 0

    def test_default_fallback(self, config):
        assert config.get("nonexistent", "key", default="ok") == "ok"

    def test_formula_weights_sum(self, config):
        formula = config.get("burnout_formula")
        total = sum(formula.values())
        assert abs(total - 1.0) < 1e-6, f"Formula weights sum to {total}, expected 1.0"
