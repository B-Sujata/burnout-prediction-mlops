"""
LLM-Powered Burnout Advisor
Uses LangChain to generate personalised, empathetic burnout advice
based on the ML model's prediction output.

Supports both OpenAI (GPT-4o) and Anthropic (Claude) backends,
switchable via configs/advisor_config.yaml.
"""

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory — builds the correct LangChain chat model from config
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm(provider: str, model: str, temperature: float, max_tokens: int):
    """
    Instantiate the correct LangChain LLM based on provider string.
    Raises clear errors if the API key is missing.
    """
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Run: pip install langchain-openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable not set.\n"
                "Export it with: export OPENAI_API_KEY='sk-...'"
            )
        logger.info(f"Using OpenAI backend: {model}")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Run: pip install langchain-anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Export it with: export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        logger.info(f"Using Anthropic backend: {model}")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            "Set advisor.provider to 'openai' or 'anthropic' in configs/advisor_config.yaml"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stress Driver Detection
# ─────────────────────────────────────────────────────────────────────────────

def identify_stress_drivers(
    features: dict,
    advisor_cfg: dict,
    n: int = 3,
) -> list[dict]:
    """
    Identify the top N features contributing most to stress.

    Each feature is normalised to [0, 1] using its known scale.
    For 'inverted' features (where lower = more stress), the normalised
    value is flipped so that a high 'stress contribution' always means
    the feature is pushing burnout up.

    Returns:
        List of dicts: [{name, label, raw_value, stress_contribution, direction}, ...]
    """
    inverted   = set(advisor_cfg.get("inverted_features", []))
    max_vals   = advisor_cfg.get("feature_max", {})
    min_vals   = advisor_cfg.get("feature_min", {})
    labels     = advisor_cfg.get("feature_labels", {})

    scored = []
    for feat, raw_val in features.items():
        max_v = max_vals.get(feat, 10)
        min_v = min_vals.get(feat, 0)
        span  = max_v - min_v
        if span == 0:
            continue

        # Normalise to [0, 1]
        norm = (float(raw_val) - min_v) / span
        norm = max(0.0, min(1.0, norm))

        # For inverted features, flip so contribution = how bad it is
        contribution = (1.0 - norm) if feat in inverted else norm

        scored.append({
            "name":               feat,
            "label":              labels.get(feat, feat.replace("_", " ").title()),
            "raw_value":          raw_val,
            "stress_contribution": round(contribution, 3),
            "direction":          "low" if feat in inverted else "high",
        })

    # Sort by contribution descending, take top N
    scored.sort(key=lambda x: x["stress_contribution"], reverse=True)
    return scored[:n]


def format_stress_drivers(drivers: list[dict]) -> str:
    """Format driver list into a readable prompt string."""
    lines = []
    for i, d in enumerate(drivers, 1):
        direction = "too low" if d["direction"] == "low" else "elevated"
        lines.append(
            f"  {i}. {d['label']} = {d['raw_value']} "
            f"({direction}, stress contribution: {d['stress_contribution']:.0%})"
        )
    return "\n".join(lines)


def format_feature_summary(features: dict, labels: dict) -> str:
    """Format all features into a readable two-column summary."""
    lines = []
    for feat, val in features.items():
        label = labels.get(feat, feat.replace("_", " ").title())
        lines.append(f"  - {label}: {val}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main Advisor Class
# ─────────────────────────────────────────────────────────────────────────────

class BurnoutAdvisor:
    """
    LangChain-powered personalised burnout advisor.

    Usage:
        advisor = BurnoutAdvisor()
        result  = advisor.advise(prediction_result)
        print(result["advice"])
    """

    def __init__(self, config_path: str = "configs/advisor_config.yaml"):
        self.config_path = config_path
        self._load_config()
        self._build_chain()

    def _load_config(self):
        """Load advisor config from YAML."""
        import yaml
        from pathlib import Path
        path = Path(self.config_path)
        if not path.exists():
            raise FileNotFoundError(f"Advisor config not found: {self.config_path}")
        with open(path) as f:
            full = yaml.safe_load(f)
        self.cfg = full["advisor"]
        logger.info(f"Advisor config loaded: provider={self.cfg['provider']}, "
                    f"model={self.cfg.get(self.cfg['provider'] + '_model')}")

    def _build_chain(self):
        """Build the LangChain chain: prompt | llm."""
        from src.advisor.prompt_templates import build_advisor_prompt

        provider   = self.cfg["provider"]
        model_key  = f"{provider}_model"
        model      = self.cfg[model_key]
        temp       = self.cfg.get("temperature", 0.7)
        max_tok    = self.cfg.get("max_tokens", 1024)
        n_drivers  = self.cfg.get("top_stress_drivers", 3)
        persona    = self.cfg.get("persona", "You are a helpful student counsellor.")

        llm    = _build_llm(provider, model, temp, max_tok)
        prompt = build_advisor_prompt(persona=persona, n_drivers=n_drivers)

        # LangChain LCEL chain: prompt → LLM
        self.chain     = prompt | llm
        self.n_drivers = n_drivers
        logger.info("LangChain advisor chain built successfully")

    def advise(self, prediction: dict) -> dict:
        """
        Generate personalised burnout advice for a student.

        Args:
            prediction: Output dict from predict_burnout() containing:
                        burnout_score, risk_level, icon, input_features,
                        and optionally the composite index values.

        Returns:
            Dict with 'advice' (str), 'stress_drivers' (list), 'provider' (str)
        """
        burnout_score = prediction["burnout_score"]
        risk_level    = prediction["risk_level"]
        features      = prediction["input_features"]

        logger.info(f"Generating LLM advice for score={burnout_score}, risk={risk_level}")

        # Identify top stress drivers
        drivers         = identify_stress_drivers(features, self.cfg, n=self.n_drivers)
        drivers_text    = format_stress_drivers(drivers)
        feature_summary = format_feature_summary(features, self.cfg.get("feature_labels", {}))

        # Get composite indices if available in prediction (added by predict_model)
        emotional_strain  = prediction.get("emotional_strain",  0.5)
        physical_stress   = prediction.get("physical_stress",   0.5)
        academic_pressure = prediction.get("academic_pressure", 0.5)
        social_stress     = prediction.get("social_stress",     0.5)
        recovery_index    = prediction.get("recovery_index",    0.5)

        # Invoke the LangChain chain
        try:
            response = self.chain.invoke({
                "burnout_score":    burnout_score,
                "risk_level":       risk_level,
                "feature_summary":  feature_summary,
                "stress_drivers":   drivers_text,
                "emotional_strain": emotional_strain,
                "physical_stress":  physical_stress,
                "academic_pressure":academic_pressure,
                "social_stress":    social_stress,
                "recovery_index":   recovery_index,
            })

            # LangChain returns an AIMessage — extract the text
            advice_text = response.content if hasattr(response, "content") else str(response)
            logger.info("LLM advice generated successfully")

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            advice_text = self._fallback_advice(burnout_score, risk_level, drivers)

        return {
            "advice":         advice_text,
            "stress_drivers": drivers,
            "provider":       self.cfg["provider"],
            "model":          self.cfg.get(f"{self.cfg['provider']}_model"),
        }

    def _fallback_advice(self, score: float, risk: str, drivers: list) -> str:
        """
        Static fallback advice shown if the LLM call fails
        (e.g. no API key set, network error).
        """
        driver_names = [d["label"] for d in drivers]
        base = {
            "Low":      "Your stress levels are manageable. Keep prioritising sleep, social connection, and physical activity.",
            "Moderate": "You're showing signs of moderate burnout. Focus on reducing your top stressors and improving your recovery habits.",
            "High":     "Your burnout risk is high. Please speak with a counsellor or trusted person as soon as possible. Reducing workload and improving sleep are critical first steps.",
        }
        tip = base.get(risk, base["Moderate"])
        drivers_str = ", ".join(driver_names) if driver_names else "several factors"
        return (
            f"[Note: LLM advice unavailable — showing static fallback]\n\n"
            f"{tip}\n\n"
            f"Your main stress contributors appear to be: {drivers_str}.\n"
            f"Burnout Score: {score}/100 | Risk: {risk}"
        )
