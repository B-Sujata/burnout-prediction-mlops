"""
Stage 8: Prediction Pipeline — Real Dataset Edition
Predicts burnout score for a new student using all 20 real features.
"""

import os, sys, json, pickle, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.utils.logger import get_logger, log_separator
from src.utils.config_loader import load_config

logger = get_logger(__name__)


def classify_risk(score: float, low: int = 33, mod: int = 66) -> dict:
    if score <= low:
        return {"risk_level": "Low",      "icon": "🟢",
                "advice": "Your stress indicators are in a healthy range. Keep up your current lifestyle habits!"}
    elif score <= mod:
        return {"risk_level": "Moderate", "icon": "🟡",
                "advice": "Moderate burnout detected. Focus on improving sleep, physical activity, and seeking social support."}
    else:
        return {"risk_level": "High",     "icon": "🔴",
                "advice": "High burnout risk! Speak with a counsellor, reduce your workload, and prioritise self-care immediately."}


def _norm(val, max_val, min_val=0.0):
    return float(np.clip((val - min_val) / (max_val - min_val), 0, 1))

def _inv(v):
    return 1.0 - v


def reconstruct_features(raw: dict, config) -> dict:
    """Reconstruct all engineered features from raw input."""
    fw      = config.get("feature_weights")
    formula = config.get("burnout_formula")

    # ── Composite features ───────────────────────────────────────────────
    es = (
        fw["emotional_strain"]["anxiety_level"]         * _norm(raw["anxiety_level"], 21)
      + fw["emotional_strain"]["depression"]            * _norm(raw["depression"],    27)
      + fw["emotional_strain"]["mental_health_history"] * _norm(raw["mental_health_history"], 1)
      + fw["emotional_strain"]["self_esteem_inv"]       * _inv(_norm(raw["self_esteem"], 30))
    )
    ps = (
        fw["physical_stress"]["headache"]           * _norm(raw["headache"],          5)
      + fw["physical_stress"]["blood_pressure"]     * _norm(raw["blood_pressure"],    3, 1)
      + fw["physical_stress"]["breathing_problem"]  * _norm(raw["breathing_problem"], 5)
    )
    ap = (
        fw["academic_pressure"]["study_load"]                       * _norm(raw["study_load"],                5)
      + fw["academic_pressure"]["future_career_concerns"]           * _norm(raw["future_career_concerns"],    5)
      + fw["academic_pressure"]["academic_performance_inv"]         * _inv(_norm(raw["academic_performance"], 5))
      + fw["academic_pressure"]["teacher_student_relationship_inv"] * _inv(_norm(raw["teacher_student_relationship"], 5))
    )
    ss = (
        fw["social_stress"]["peer_pressure"]         * _norm(raw["peer_pressure"],    5)
      + fw["social_stress"]["bullying"]              * _norm(raw["bullying"],          5)
      + fw["social_stress"]["noise_level"]           * _norm(raw["noise_level"],       5)
      + fw["social_stress"]["living_conditions_inv"] * _inv(_norm(raw["living_conditions"], 5))
      + fw["social_stress"]["safety_inv"]            * _inv(_norm(raw["safety"],       5))
      + fw["social_stress"]["basic_needs_inv"]       * _inv(_norm(raw["basic_needs"],  5))
    )
    ri = (
        fw["recovery_index"]["sleep_quality"]              * _norm(raw["sleep_quality"],             5)
      + fw["recovery_index"]["social_support"]             * _norm(raw["social_support"],            3)
      + fw["recovery_index"]["extracurricular_activities"] * _norm(raw["extracurricular_activities"],5)
    )

    # ── Interaction features ─────────────────────────────────────────────
    stress_sleep   = raw["anxiety_level"] / (raw["sleep_quality"] + 1e-6)
    acad_burden    = (raw["study_load"] + raw["future_career_concerns"]) / 10
    isolation      = (5 - _norm(raw["social_support"],3)*5) * (raw["peer_pressure"]+1) / 6
    env_quality    = (raw["living_conditions"] + raw["safety"] + raw["basic_needs"]) / 15
    mental_load    = (_norm(raw["anxiety_level"],21) + _norm(raw["depression"],27)) / 2

    return {
        **raw,
        "emotional_strain":   round(es,    4),
        "physical_stress":    round(ps,    4),
        "academic_pressure":  round(ap,    4),
        "social_stress":      round(ss,    4),
        "recovery_index":     round(ri,    4),
        "stress_sleep_ratio": round(stress_sleep, 3),
        "academic_burden":    round(acad_burden,  3),
        "isolation_index":    round(isolation,    3),
        "environment_quality":round(env_quality,  3),
        "mental_load":        round(mental_load,  3),
    }


def predict_burnout(
    raw_input: dict,
    config_path: str = "configs/config.yaml",
    use_advisor: bool = False,
    advisor_config_path: str = "configs/advisor_config.yaml",
) -> dict:
    log_separator(logger, "Prediction Pipeline")
    config = load_config(config_path)

    best_model_path    = config.get("paths", "best_model")
    feature_names_path = os.path.join(config.get("paths", "model_dir"), "feature_names.json")
    low_t = config.get("burnout_thresholds", "low",      default=33)
    mod_t = config.get("burnout_thresholds", "moderate", default=66)

    with open(best_model_path,     "rb") as f: pipeline      = pickle.load(f)
    with open(feature_names_path        ) as f: feature_names = json.load(f)

    full  = reconstruct_features(raw_input, config)
    X     = np.array([[full.get(feat, 0.0) for feat in feature_names]])

    score = float(np.clip(pipeline.predict(X)[0], 0, 100))
    score = round(score, 2)
    risk  = classify_risk(score, low_t, mod_t)

    # Include composite indices in result (used by the LLM advisor)
    result = {
        "burnout_score":    score,
        **risk,
        "input_features":   raw_input,
        # Expose engineered composite indices for the advisor
        "emotional_strain":  round(full.get("emotional_strain",  0.0), 4),
        "physical_stress":   round(full.get("physical_stress",   0.0), 4),
        "academic_pressure": round(full.get("academic_pressure", 0.0), 4),
        "social_stress":     round(full.get("social_stress",     0.0), 4),
        "recovery_index":    round(full.get("recovery_index",    0.0), 4),
    }
    logger.info(f"Score: {score}/100  |  {risk['icon']} {risk['risk_level']}")

    # ── Optional: LLM Advisor ──────────────────────────────────────────────
    if use_advisor:
        try:
            from src.advisor.burnout_advisor import BurnoutAdvisor
            advisor         = BurnoutAdvisor(config_path=advisor_config_path)
            advisor_output  = advisor.advise(result)
            result["llm_advice"]      = advisor_output["advice"]
            result["stress_drivers"]  = advisor_output["stress_drivers"]
            result["llm_provider"]    = advisor_output["provider"]
            result["llm_model"]       = advisor_output["model"]
            logger.info(f"LLM advice generated via {advisor_output['provider']}")
        except Exception as e:
            logger.warning(f"Advisor unavailable: {e}")
            result["llm_advice"] = None

    return result


PROMPTS = {
    "anxiety_level":              ("Anxiety Level",               0, 21),
    "self_esteem":                ("Self Esteem",                 0, 30),
    "mental_health_history":      ("Mental Health History (0=No, 1=Yes)", 0, 1),
    "depression":                 ("Depression Score",            0, 27),
    "headache":                   ("Headache Frequency",          0,  5),
    "blood_pressure":             ("Blood Pressure (1=Normal, 2=Elevated, 3=High)", 1, 3),
    "sleep_quality":              ("Sleep Quality",               0,  5),
    "breathing_problem":          ("Breathing Problems",          0,  5),
    "noise_level":                ("Noise Level (in study area)", 0,  5),
    "living_conditions":          ("Living Conditions Quality",   0,  5),
    "safety":                     ("Sense of Safety",             0,  5),
    "basic_needs":                ("Basic Needs Met",             0,  5),
    "academic_performance":       ("Academic Performance",        0,  5),
    "study_load":                 ("Study Load",                  0,  5),
    "teacher_student_relationship":("Teacher-Student Relationship",0, 5),
    "future_career_concerns":     ("Future Career Concerns",      0,  5),
    "social_support":             ("Social Support",              0,  3),
    "peer_pressure":              ("Peer Pressure",               0,  5),
    "extracurricular_activities": ("Extracurricular Activities",  0,  5),
    "bullying":                   ("Bullying Experienced",        0,  5),
}

def interactive_prediction(use_advisor: bool = False):
    """Run the full assessment using the natural language questionnaire."""
    
    user_input = collect_answers_cli()

    print("  ⏳ Calculating your burnout score...\n")
    result = predict_burnout(user_input, use_advisor=use_advisor)

    # ── Score display ──────────────────────────────────────────────────
    score = result["burnout_score"]
    risk  = result["risk_level"]
    icon  = result["icon"]

    bar_filled = int(score / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)

    print("═"*65)
    print(f"  {icon}  YOUR BURNOUT SCORE")
    print(f"\n     [{bar}]  {score}/100")
    print(f"\n     RISK LEVEL: {risk}")
    print(f"\n  💡 {result['advice']}")

    # ── Composite index breakdown ──────────────────────────────────────
    print("\n  📊 Breakdown:")
    indices = [
        ("Emotional Strain",  result.get("emotional_strain",  0)),
        ("Physical Stress",   result.get("physical_stress",   0)),
        ("Academic Pressure", result.get("academic_pressure", 0)),
        ("Social Stress",     result.get("social_stress",     0)),
        ("Recovery Index",    result.get("recovery_index",    0)),
    ]
    for name, val in indices:
        filled = int(val * 10)
        mini   = "▓" * filled + "░" * (10 - filled)
        arrow  = "↓ good" if name == "Recovery Index" else ""
        print(f"     {name:20s} [{mini}]  {val:.2f}  {arrow}")

    # ── LLM advice ────────────────────────────────────────────────────
    if use_advisor and result.get("llm_advice"):
        print("\n" + "─"*65)
        print("  🤖 PERSONALISED ADVICE FROM AI COUNSELLOR")
        print("─"*65)
        print(f"\n{result['llm_advice']}")

        if result.get("stress_drivers"):
            print("\n  📌 Your Top Stress Drivers:")
            for d in result["stress_drivers"]:
                print(f"     • {d['label']} (contribution: {d['stress_contribution']:.0%})")

    print("═"*65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Burnout Prediction")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive CLI prediction")
    parser.add_argument("--json-input",  type=str,
                        help="JSON string of input features")
    parser.add_argument("--advise",      action="store_true",
                        help="Enable LLM-powered personalised advice (requires API key)")
    args = parser.parse_args()

    if args.json_input:
        result = predict_burnout(json.loads(args.json_input), use_advisor=args.advise)
        # Remove non-serialisable keys for clean JSON output
        output = {k: v for k, v in result.items()
                  if k not in ("stress_drivers",) or isinstance(v, (str, int, float, bool, type(None)))}
        print(json.dumps(result, indent=2, default=str))
    else:
        interactive_prediction(use_advisor=args.advise)