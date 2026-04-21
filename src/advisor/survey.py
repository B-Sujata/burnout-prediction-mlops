"""
Survey Engine
Presents human-readable questions to the student, collects labeled answers,
and maps them to the numeric values the ML model expects.

The student never sees a raw number. They choose from descriptive options.
"""

import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Survey Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_survey(survey_path: str = "configs/survey_questions.yaml") -> list:
    """Load survey questions from YAML."""
    path = Path(survey_path)
    if not path.exists():
        raise FileNotFoundError(f"Survey config not found: {survey_path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    questions = data["questions"]
    logger.info(f"Survey loaded: {len(questions)} questions")
    return questions


def group_by_section(questions: list) -> dict:
    """Group questions by their section label."""
    sections = {}
    for q in questions:
        section = q.get("section", "General")
        sections.setdefault(section, []).append(q)
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# CLI Survey Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_cli_survey(survey_path: str = "configs/survey_questions.yaml") -> dict:
    """
    Run the survey in the terminal.

    Presents each question with numbered options.
    The student types a number (1, 2, 3…) and never sees a raw model value.

    Returns:
        dict mapping feature_id → numeric value (ready for predict_burnout)
    """
    questions = load_survey(survey_path)
    sections  = group_by_section(questions)
    answers   = {}
    total     = len(questions)
    answered  = 0

    print("\n" + "═" * 65)
    print("  🎓 Student Wellbeing Assessment")
    print("  Answer honestly — there are no right or wrong answers.")
    print("  This takes about 3–5 minutes.")
    print("═" * 65)

    for section_name, qs in sections.items():
        print(f"\n  {'─'*60}")
        print(f"  {section_name}")
        print(f"  {'─'*60}")

        for q in qs:
            answered += 1
            options   = q["options"]
            n_opts    = len(options)

            print(f"\n  [{answered}/{total}]  {q['question']}\n")
            for i, opt in enumerate(options, 1):
                print(f"    {i}.  {opt['label']}")

            # Input validation loop
            while True:
                try:
                    raw = input(f"\n  Your answer (1–{n_opts}): ").strip()
                    choice = int(raw)
                    if 1 <= choice <= n_opts:
                        selected = options[choice - 1]
                        answers[q["id"]] = selected["value"]
                        logger.debug(
                            f"  {q['id']}: choice={choice} "
                            f"→ value={selected['value']} "
                            f"('{selected['label'][:40]}')"
                        )
                        break
                    else:
                        print(f"  ⚠  Please enter a number between 1 and {n_opts}.")
                except (ValueError, KeyboardInterrupt):
                    print(f"  ⚠  Please enter a number between 1 and {n_opts}.")

    print("\n" + "═" * 65)
    print("  ✅  Assessment complete. Analysing your responses...")
    print("═" * 65 + "\n")

    logger.info(f"Survey completed. Collected {len(answers)} feature values.")
    return answers


# ─────────────────────────────────────────────────────────────────────────────
# JSON-mode: convert a dict of labeled answers to numeric values
# (used when the survey is completed via a web form)
# ─────────────────────────────────────────────────────────────────────────────

def labeled_to_numeric(
    labeled_answers: dict,
    survey_path: str = "configs/survey_questions.yaml",
) -> dict:
    """
    Convert a dict of {feature_id: option_index (1-based)} to
    {feature_id: numeric_value} for the model.

    Args:
        labeled_answers: {feature_id: 1-based option index chosen by student}
        survey_path: path to survey YAML

    Returns:
        {feature_id: numeric model value}
    """
    questions = load_survey(survey_path)
    q_map = {q["id"]: q for q in questions}
    numeric = {}

    for feat_id, choice_idx in labeled_answers.items():
        if feat_id not in q_map:
            logger.warning(f"Unknown feature in labeled answers: {feat_id}")
            continue
        options = q_map[feat_id]["options"]
        idx = int(choice_idx) - 1  # convert to 0-based
        if 0 <= idx < len(options):
            numeric[feat_id] = options[idx]["value"]
        else:
            logger.warning(f"{feat_id}: invalid option index {choice_idx}")

    return numeric


def get_survey_schema(survey_path: str = "configs/survey_questions.yaml") -> list:
    """
    Return a clean schema of questions + options for building web forms.
    Strips out the raw 'value' field — only the labels are exposed to the frontend.
    """
    questions = load_survey(survey_path)
    schema = []
    for q in questions:
        schema.append({
            "id":       q["id"],
            "section":  q.get("section", "General"),
            "question": q["question"],
            "options":  [opt["label"] for opt in q["options"]],  # labels only
        })
    return schema


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from src.models.predict_model import predict_burnout
    import src.utils.config_loader as cl
    cl.ConfigLoader._instance = None

    # Run the survey
    numeric_answers = run_cli_survey()

    # Predict
    result = predict_burnout(numeric_answers)

    print(f"  {result['icon']}  BURNOUT SCORE : {result['burnout_score']} / 100")
    print(f"      RISK LEVEL  : {result['risk_level']}")
    print(f"\n  💡  {result['advice']}\n")
