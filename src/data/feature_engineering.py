"""
Stage 3 & 4: Feature Engineering + Burnout Score Construction
Adapted for StressLevelDataset — real Kaggle dataset.

All features are normalised to [0,1] using their known max values before
being combined into composite indices and the final burnout score.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger, log_separator
from src.utils.config_loader import load_config

logger = get_logger(__name__)


# ── Normalisation helpers ─────────────────────────────────────────────────────

def norm(series: pd.Series, max_val: float, min_val: float = 0.0) -> pd.Series:
    """Normalise a series to [0, 1] given known min/max."""
    return ((series - min_val) / (max_val - min_val)).clip(0, 1)

def inv(series: pd.Series) -> pd.Series:
    """Invert a [0,1] series: high → low, low → high."""
    return 1.0 - series


# ── Composite feature builders ────────────────────────────────────────────────

def build_emotional_strain(df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Emotional Strain — psychological distress composite.
      anxiety_level   (0–21)  higher = worse
      depression      (0–27)  higher = worse
      mental_health_history (0–1) 1 = has history
      self_esteem_inv (0–30)  low self-esteem = higher strain
    """
    anxiety    = norm(df["anxiety_level"],         21)
    depression = norm(df["depression"],             27)
    mhh        = norm(df["mental_health_history"],   1)
    se_inv     = inv(norm(df["self_esteem"],         30))

    return (
        weights["anxiety_level"]      * anxiety
      + weights["depression"]         * depression
      + weights["mental_health_history"] * mhh
      + weights["self_esteem_inv"]    * se_inv
    ).rename("emotional_strain")


def build_physical_stress(df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Physical Stress — somatic symptoms of stress.
      headache          (0–5)  higher = more headaches
      blood_pressure    (1–3)  shifted to [0,1]
      breathing_problem (0–5)  higher = worse
    """
    headache   = norm(df["headache"],           5)
    bp         = norm(df["blood_pressure"],     3, min_val=1)   # scale is 1–3
    breathing  = norm(df["breathing_problem"],  5)

    return (
        weights["headache"]           * headache
      + weights["blood_pressure"]     * bp
      + weights["breathing_problem"]  * breathing
    ).rename("physical_stress")


def build_academic_pressure(df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Academic Pressure — stress from academic demands.
      study_load                    (0–5) higher = more load
      future_career_concerns        (0–5) higher = more anxiety
      academic_performance_inv      (0–5) low performance → higher pressure
      teacher_student_relationship_inv (0–5) poor relationship → higher pressure
    """
    study_load   = norm(df["study_load"],               5)
    career       = norm(df["future_career_concerns"],   5)
    perf_inv     = inv(norm(df["academic_performance"], 5))
    tsr_inv      = inv(norm(df["teacher_student_relationship"], 5))

    return (
        weights["study_load"]                       * study_load
      + weights["future_career_concerns"]           * career
      + weights["academic_performance_inv"]         * perf_inv
      + weights["teacher_student_relationship_inv"] * tsr_inv
    ).rename("academic_pressure")


def build_social_stress(df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Social Stress — environmental and relational stressors.
      peer_pressure         (0–5) higher = more pressure
      bullying              (0–5) higher = more bullying
      noise_level           (0–5) higher = more disruptive
      living_conditions_inv (0–5) poor conditions → higher stress
      safety_inv            (0–5) lower safety → higher stress
      basic_needs_inv       (0–5) unmet needs → higher stress
    """
    peer    = norm(df["peer_pressure"],     5)
    bully   = norm(df["bullying"],          5)
    noise   = norm(df["noise_level"],       5)
    lc_inv  = inv(norm(df["living_conditions"], 5))
    safe_inv= inv(norm(df["safety"],        5))
    bn_inv  = inv(norm(df["basic_needs"],   5))

    return (
        weights["peer_pressure"]        * peer
      + weights["bullying"]             * bully
      + weights["noise_level"]          * noise
      + weights["living_conditions_inv"]* lc_inv
      + weights["safety_inv"]           * safe_inv
      + weights["basic_needs_inv"]      * bn_inv
    ).rename("social_stress")


def build_recovery_index(df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Recovery Index — protective / restorative factors.
    Higher recovery → lower burnout (used as penalty in formula).
      sleep_quality            (0–5) higher = better sleep
      social_support           (0–3) higher = more support
      extracurricular_activities (0–5) higher = more healthy activities
    """
    sleep   = norm(df["sleep_quality"],              5)
    support = norm(df["social_support"],             3)
    extra   = norm(df["extracurricular_activities"], 5)

    return (
        weights["sleep_quality"]              * sleep
      + weights["social_support"]             * support
      + weights["extracurricular_activities"] * extra
    ).rename("recovery_index")


# ── Burnout Score ─────────────────────────────────────────────────────────────

def compute_burnout_score(df: pd.DataFrame, formula: dict) -> pd.Series:
    """
    Burnout Score (raw) =
        w_es * emotional_strain
      + w_ps * physical_stress
      + w_ap * academic_pressure
      + w_ss * social_stress
      - w_ri * recovery_index     ← recovery REDUCES burnout

    Min-Max normalised to [0, 100].
    """
    w_es = formula["emotional_strain_weight"]
    w_ps = formula["physical_stress_weight"]
    w_ap = formula["academic_pressure_weight"]
    w_ss = formula["social_stress_weight"]
    w_ri = formula["recovery_penalty_weight"]

    raw = (
        w_es * df["emotional_strain"]
      + w_ps * df["physical_stress"]
      + w_ap * df["academic_pressure"]
      + w_ss * df["social_stress"]
      - w_ri * df["recovery_index"]
    )

    mn, mx = raw.min(), raw.max()
    normalised = ((raw - mn) / (mx - mn) * 100) if mx > mn else pd.Series(np.zeros(len(raw)))
    return normalised.round(2).rename("burnout_score")


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extra interaction features derived from real columns."""
    # Stress amplified by poor sleep
    df["stress_sleep_ratio"]   = (df["anxiety_level"] / (df["sleep_quality"] + 1e-6)).round(3)
    # Combined academic burden
    df["academic_burden"]      = ((df["study_load"] + df["future_career_concerns"]) / 10).round(3)
    # Social isolation index
    df["isolation_index"]      = ((5 - df["social_support"].clip(0,3)/3*5) *
                                   (df["peer_pressure"] + 1) / 6).round(3)
    # Environmental quality (higher = better environment)
    df["environment_quality"]  = ((df["living_conditions"] + df["safety"] + df["basic_needs"]) / 15).round(3)
    # Mental load (depression + anxiety combined, normalised)
    df["mental_load"]          = ((df["anxiety_level"]/21 + df["depression"]/27) / 2).round(3)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def engineer_features(config_path: str = "configs/config.yaml") -> str:
    log_separator(logger, "Stage 3 & 4: Feature Engineering + Burnout Score")
    config = load_config(config_path)

    processed_path  = config.get("paths", "processed_data")
    engineered_path = config.get("paths", "engineered_data")
    fw              = config.get("feature_weights")
    formula         = config.get("burnout_formula")

    df = pd.read_csv(processed_path)
    logger.info(f"Loaded processed data: {df.shape}")

    # ── Composite features ───────────────────────────────────────────────
    df["emotional_strain"]  = build_emotional_strain(df,  fw["emotional_strain"])
    df["physical_stress"]   = build_physical_stress(df,   fw["physical_stress"])
    df["academic_pressure"] = build_academic_pressure(df, fw["academic_pressure"])
    df["social_stress"]     = build_social_stress(df,     fw["social_stress"])
    df["recovery_index"]    = build_recovery_index(df,    fw["recovery_index"])

    logger.info("Composite features created: emotional_strain, physical_stress, "
                "academic_pressure, social_stress, recovery_index")

    # ── Interaction features ─────────────────────────────────────────────
    df = add_interaction_features(df)
    logger.info("Interaction features: stress_sleep_ratio, academic_burden, "
                "isolation_index, environment_quality, mental_load")

    # ── Burnout score ────────────────────────────────────────────────────
    df["burnout_score"] = compute_burnout_score(df, formula)
    logger.info(f"\nBurnout Score statistics:\n{df['burnout_score'].describe().round(2)}")

    # ── Validate alignment with original stress_level ────────────────────
    if "stress_level" in df.columns:
        logger.info("\nMean burnout score per original stress_level (validation):")
        for lvl, label in [(0,"Low"), (1,"Moderate"), (2,"High")]:
            mean_score = df[df["stress_level"]==lvl]["burnout_score"].mean()
            logger.info(f"  stress_level={lvl} ({label:8s}): burnout_score mean = {mean_score:.2f}")

    Path(os.path.dirname(engineered_path)).mkdir(parents=True, exist_ok=True)
    df.to_csv(engineered_path, index=False)
    logger.info(f"Engineered dataset saved → {engineered_path}  shape={df.shape}")
    return engineered_path

if __name__ == "__main__":
    engineer_features()
