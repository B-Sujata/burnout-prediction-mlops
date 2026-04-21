"""
Student Burnout Questionnaire — Human-Readable CLI
===================================================
20 plain-English questions with descriptive multiple-choice options.
Each answer maps silently to a numeric feature value.
The user never sees a number.

Designed following the principles of validated clinical screening tools
(PHQ-9, GAD-7) — descriptive anchors, no numeric scales shown.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Question Bank
#
# Each entry:
#   "feature"   → the model's internal feature name
#   "question"  → what the student sees
#   "section"   → groups related questions visually
#   "options"   → list of (display_text, numeric_value) tuples
#                 ordered from least to most stressful
# ─────────────────────────────────────────────────────────────────────────────

QUESTIONS = [

    # ── Section 1: How you've been feeling ──────────────────────────────────

    {
        "feature":  "anxiety_level",
        "section":  "How You've Been Feeling Lately",
        "question": "How often do you feel anxious, nervous, or on edge?",
        "options": [
            ("Rarely or never — I feel pretty calm most of the time",          2),
            ("Occasionally — maybe a few times a month",                        6),
            ("Regularly — several times a week, it's becoming noticeable",     11),
            ("Very often — almost every day I feel wound up or worried",       16),
            ("Constantly — I feel anxious nearly all the time",                21),
        ],
    },

    {
        "feature":  "depression",
        "section":  None,   # same section as above
        "question": "How often do you feel low, hopeless, or have little interest in things you used to enjoy?",
        "options": [
            ("Not at all — I feel generally positive or okay",                  2),
            ("Occasionally — I have some low days but bounce back",             7),
            ("Often — I frequently feel flat, empty, or unmotivated",          14),
            ("Most of the time — I struggle to feel good about much",          20),
            ("Nearly every day — it's affecting my daily life significantly",  27),
        ],
    },

    {
        "feature":  "mental_fatigue",
        "section":  None,
        "question": "How mentally exhausted do you feel by the end of a typical day?",
        "options": [
            ("Not really — I still have energy and mental clarity",             1),
            ("A little — I feel tired but it's manageable",                     3),
            ("Quite a lot — my brain feels foggy and I struggle to focus",      6),
            ("Very much — I feel completely drained most evenings",             8),
            ("Totally depleted — I can barely think straight by midday",       10),
        ],
    },

    {
        "feature":  "self_esteem",
        "section":  None,
        "question": "How do you feel about yourself and your abilities overall?",
        "options": [
            ("Very confident — I feel good about who I am and what I can do",  28),
            ("Generally positive — I have some doubts but mostly feel capable", 21),
            ("Mixed — I often question myself and compare myself to others",    14),
            ("Low — I frequently feel inadequate or not good enough",           7),
            ("Very low — I rarely feel capable or worthy",                      2),
        ],
    },

    {
        "feature":  "mental_health_history",
        "section":  None,
        "question": "Have you previously been diagnosed with or treated for a mental health condition (e.g. anxiety, depression)?",
        "options": [
            ("No — I have no history of mental health diagnoses or treatment",  0),
            ("Yes — I have a history of mental health diagnosis or treatment",  1),
        ],
    },

    # ── Section 2: Your body ─────────────────────────────────────────────────

    {
        "feature":  "sleep_quality",
        "section":  "Your Body & Physical Health",
        "question": "How would you describe the quality of your sleep most nights?",
        "options": [
            ("Excellent — I fall asleep easily, sleep through, and wake refreshed",  5),
            ("Good — mostly restful with occasional off nights",                      4),
            ("Fair — I wake up a few times or don't feel fully rested",              3),
            ("Poor — I struggle to fall asleep or wake up exhausted",                1),
            ("Very poor — my sleep is consistently broken or minimal",               0),
        ],
    },

    {
        "feature":  "headache",
        "section":  None,
        "question": "How frequently do you get headaches?",
        "options": [
            ("Rarely or never",                                                  0),
            ("Once or twice a month",                                            1),
            ("Once or twice a week",                                             2),
            ("Several times a week",                                             3),
            ("Daily or almost daily",                                            5),
        ],
    },

    {
        "feature":  "blood_pressure",
        "section":  None,
        "question": "What is your blood pressure situation? (If unsure, choose the first option)",
        "options": [
            ("Normal / I've never been told I have high blood pressure",         1),
            ("Slightly elevated — I've been told to monitor it",                 2),
            ("High — I've been diagnosed with hypertension or it's a concern",  3),
        ],
    },

    {
        "feature":  "breathing_problem",
        "section":  None,
        "question": "Do you experience breathing difficulties, chest tightness, or shortness of breath?",
        "options": [
            ("Never — I breathe easily without any issues",                      0),
            ("Rarely — only in very specific situations (exercise, illness)",    1),
            ("Sometimes — a few times a week, mild tightness or breathlessness", 2),
            ("Often — it happens regularly and is noticeable",                   3),
            ("Frequently — it's a significant and recurring problem",            5),
        ],
    },

    # ── Section 3: Academic life ─────────────────────────────────────────────

    {
        "feature":  "study_load",
        "section":  "Your Academic Life",
        "question": "How would you describe your current study workload?",
        "options": [
            ("Light — I have plenty of free time and coursework feels manageable",    1),
            ("Moderate — it keeps me busy but I can handle it",                       2),
            ("Heavy — I often feel behind or overwhelmed with the volume",            3),
            ("Very heavy — I barely have time for anything other than studying",      4),
            ("Crushing — the workload is beyond what I can reasonably handle",        5),
        ],
    },

    {
        "feature":  "academic_performance",
        "section":  None,
        "question": "How are you doing academically right now?",
        "options": [
            ("Excellent — performing at the top of my ability or class",              5),
            ("Good — meeting expectations and feeling satisfied with results",        4),
            ("Average — passing but not where I'd like to be",                        3),
            ("Struggling — grades are slipping and I'm falling behind",               2),
            ("Failing or at risk — I'm seriously concerned about my academic future", 0),
        ],
    },

    {
        "feature":  "future_career_concerns",
        "section":  None,
        "question": "How worried are you about your career prospects after graduation?",
        "options": [
            ("Not at all — I feel clear about my path and confident",                 0),
            ("Mildly — I have some uncertainty but it doesn't keep me up at night",   1),
            ("Moderately — I often wonder if I'll find work in my field",             2),
            ("Very worried — career anxiety is affecting my focus and motivation",    4),
            ("Extremely worried — this is one of my biggest sources of stress",       5),
        ],
    },

    {
        "feature":  "teacher_student_relationship",
        "section":  None,
        "question": "How would you describe your relationship with your teachers or lecturers?",
        "options": [
            ("Excellent — they are supportive, approachable, and invested in my success", 5),
            ("Good — generally positive with good communication",                          4),
            ("Neutral — professional but not particularly warm or helpful",                3),
            ("Strained — I feel dismissed, unsupported, or ignored",                      2),
            ("Very poor — there is significant conflict or indifference",                  0),
        ],
    },

    # ── Section 4: Your social world ─────────────────────────────────────────

    {
        "feature":  "social_support",
        "section":  "Your Social World",
        "question": "When you're struggling, how much genuine support do you have from friends or family?",
        "options": [
            ("Strong — I have people I can always turn to and they show up for me",  3),
            ("Some — I have a few people I can lean on when needed",                 2),
            ("Limited — I don't have many people I feel comfortable opening up to",  1),
            ("Very little — I mostly deal with things alone",                        0),
        ],
    },

    {
        "feature":  "peer_pressure",
        "section":  None,
        "question": "How much pressure do you feel from peers to perform, fit in, or keep up?",
        "options": [
            ("None — I don't feel pressured by what others think or do",             0),
            ("A little — occasional comparison but it doesn't affect me much",       1),
            ("Moderate — I regularly feel pressure to match others academically or socially", 3),
            ("A lot — peer comparison significantly affects my choices and mood",    4),
            ("Intense — I feel constant pressure to prove myself or keep up",        5),
        ],
    },

    {
        "feature":  "bullying",
        "section":  None,
        "question": "Have you experienced bullying, harassment, or exclusion from peers?",
        "options": [
            ("Never — I feel respected and included",                                0),
            ("Rarely — an isolated incident or two, nothing ongoing",                1),
            ("Sometimes — it happens occasionally and affects my mood",              2),
            ("Often — I regularly experience unkind treatment from others",          4),
            ("Severely — ongoing bullying or harassment that seriously affects me",  5),
        ],
    },

    {
        "feature":  "extracurricular_activities",
        "section":  None,
        "question": "How much time do you spend on activities you genuinely enjoy outside of studying? (Sports, clubs, hobbies, etc.)",
        "options": [
            ("A lot — I have an active life with hobbies and activities I love",     5),
            ("Some — I make time for at least one or two things I enjoy",            4),
            ("A little — I occasionally do something fun but it's rare",             2),
            ("Barely — I rarely have time or energy for anything enjoyable",         1),
            ("None — my life is study only right now",                               0),
        ],
    },

    # ── Section 5: Your environment ──────────────────────────────────────────

    {
        "feature":  "noise_level",
        "section":  "Your Living & Study Environment",
        "question": "How noisy or disruptive is your study environment?",
        "options": [
            ("Very quiet — I can concentrate easily without distraction",            0),
            ("Generally quiet — occasional noise but manageable",                    1),
            ("Somewhat noisy — it affects my concentration regularly",               2),
            ("Very noisy — I find it hard to study or relax at home",               4),
            ("Constantly disruptive — noise is a major barrier to studying",         5),
        ],
    },

    {
        "feature":  "living_conditions",
        "section":  None,
        "question": "How would you describe your overall living situation? (Space, comfort, stability)",
        "options": [
            ("Excellent — comfortable, stable, and conducive to studying",           5),
            ("Good — a few minor issues but generally fine",                         4),
            ("Fair — some problems that occasionally stress me out",                 3),
            ("Poor — my living situation causes regular stress or discomfort",       1),
            ("Very poor — my living situation is a major source of stress",          0),
        ],
    },

    {
        "feature":  "safety",
        "section":  None,
        "question": "How safe do you feel in your home and neighbourhood?",
        "options": [
            ("Very safe — I feel completely secure where I live",                    5),
            ("Mostly safe — occasional concerns but generally fine",                 4),
            ("Somewhat unsafe — I have regular concerns about my safety",            2),
            ("Unsafe — I often feel at risk or anxious about my environment",        1),
            ("Very unsafe — my safety is a serious and constant concern",            0),
        ],
    },

    {
        "feature":  "basic_needs",
        "section":  None,
        "question": "Are your basic needs being met? (Food, shelter, money for essentials)",
        "options": [
            ("Fully — I have no financial or basic needs concerns",                  5),
            ("Mostly — occasional tightness but nothing serious",                    4),
            ("Sometimes — I regularly worry about affording essentials",             2),
            ("Barely — I frequently go without things I need",                       1),
            ("Not at all — financial or basic needs stress is severe",               0),
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "blue":   "\033[94m",
    "grey":   "\033[90m",
}

def _c(text, *color_keys):
    """Wrap text in ANSI color codes."""
    codes = "".join(COLORS.get(k, "") for k in color_keys)
    return f"{codes}{text}{COLORS['reset']}"

def _clear_line():
    print("\033[A\033[K", end="")

def _print_header():
    print()
    print(_c("╔══════════════════════════════════════════════════════════╗", "cyan"))
    print(_c("║   🧠  Student Burnout Self-Assessment                    ║", "cyan", "bold"))
    print(_c("║   This takes about 3–4 minutes. Be honest with yourself. ║", "cyan"))
    print(_c("╚══════════════════════════════════════════════════════════╝", "cyan"))
    print()
    print(_c("  Use the number keys to select your answer.", "dim"))
    print(_c("  There are no right or wrong answers.", "dim"))
    print()

def _print_progress(current, total):
    filled = int((current / total) * 30)
    bar    = "█" * filled + "░" * (30 - filled)
    pct    = int((current / total) * 100)
    print(_c(f"  Progress  [{bar}] {pct}%  ({current}/{total})", "grey"))
    print()

def _print_section(name):
    print()
    print(_c(f"  ── {name} ──────────────────────────────────", "blue", "bold"))
    print()

def _ask_question(q_num, total, entry):
    """
    Display one question and collect the user's answer.
    Returns the numeric value mapped to the chosen option.
    """
    options = entry["options"]

    print(_c(f"  Q{q_num}. {entry['question']}", "bold"))
    print()

    for i, (text, _value) in enumerate(options, 1):
        print(f"    {_c(str(i), 'yellow', 'bold')}.  {text}")

    print()

    while True:
        try:
            raw = input(_c(f"  Your answer (1–{len(options)}): ", "cyan")).strip()
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                chosen_text, numeric_value = options[idx]
                print(_c(f"  ✓ Got it.", "green", "dim"))
                print()
                return numeric_value, chosen_text
            else:
                print(_c(f"  ⚠  Please enter a number between 1 and {len(options)}.", "yellow"))
        except (ValueError, KeyboardInterrupt):
            if raw.lower() in ("q", "quit", "exit"):
                print("\n  Assessment cancelled.")
                sys.exit(0)
            print(_c(f"  ⚠  Please enter a number between 1 and {len(options)}.", "yellow"))


# ─────────────────────────────────────────────────────────────────────────────
# Main questionnaire runner
# ─────────────────────────────────────────────────────────────────────────────

def run_questionnaire() -> dict:
    """
    Run the full 20-question assessment in the terminal.

    Returns:
        Dict of {feature_name: numeric_value} — ready to pass to predict_burnout()
    """
    _print_header()
    input(_c("  Press Enter to begin...", "dim"))
    print()

    answers       = {}
    answer_labels = {}   # human-readable versions for the summary
    current_section = None
    total = len(QUESTIONS)

    for i, entry in enumerate(QUESTIONS, 1):

        # Print section header when section changes
        if entry["section"] and entry["section"] != current_section:
            current_section = entry["section"]
            _print_section(current_section)

        _print_progress(i - 1, total)
        numeric_val, chosen_text = _ask_question(i, total, entry)

        answers[entry["feature"]]       = numeric_val
        answer_labels[entry["feature"]] = chosen_text

    _print_progress(total, total)
    return answers, answer_labels


# ─────────────────────────────────────────────────────────────────────────────
# Result display
# ─────────────────────────────────────────────────────────────────────────────

def display_result(result: dict, answer_labels: dict) -> None:
    """Pretty-print the prediction result after the questionnaire."""

    score      = result["burnout_score"]
    risk       = result["risk_level"]
    icon       = result["icon"]
    static_tip = result["advice"]

    # Score bar
    filled = int(score / 100 * 40)
    if score <= 33:
        bar_color = "green"
    elif score <= 66:
        bar_color = "yellow"
    else:
        bar_color = "cyan"   # terminal red isn't great, cyan stands out

    bar = _c("█" * filled, bar_color) + _c("░" * (40 - filled), "grey")

    print()
    print(_c("╔══════════════════════════════════════════════════════════╗", "cyan"))
    print(_c("║               YOUR BURNOUT ASSESSMENT RESULT             ║", "cyan", "bold"))
    print(_c("╚══════════════════════════════════════════════════════════╝", "cyan"))
    print()
    print(f"  {icon}  {_c('BURNOUT SCORE', 'bold')} :  {_c(f'{score}/100', 'bold')}")
    print(f"       [{bar}]")
    print(f"  {_c('RISK LEVEL', 'bold')}    :  {_c(risk, 'bold')}")
    print()

    # Composite index breakdown
    print(_c("  ── Stress Index Breakdown ──────────────────────────", "grey"))
    indices = [
        ("Emotional Strain",   result.get("emotional_strain",  0), False),
        ("Physical Stress",    result.get("physical_stress",   0), False),
        ("Academic Pressure",  result.get("academic_pressure", 0), False),
        ("Social Stress",      result.get("social_stress",     0), False),
        ("Recovery Index",     result.get("recovery_index",    0), True),   # True = higher is good
    ]
    for label, val, inverted in indices:
        pct   = int(val * 100)
        bar_w = int(val * 25)
        color = ("green" if inverted else "cyan") if val < 0.4 else \
                ("yellow" if val < 0.7 else ("green" if inverted else "cyan"))
        mini_bar = _c("▓" * bar_w, color) + _c("░" * (25 - bar_w), "grey")
        suffix = " ← protective" if inverted else ""
        print(f"  {label:22s}  [{mini_bar}]  {pct:3d}%{suffix}")
    print()

    # Top stress drivers (if advisor ran)
    if result.get("stress_drivers"):
        print(_c("  ── Your Top Stress Drivers ──────────────────────────", "grey"))
        for i, d in enumerate(result["stress_drivers"], 1):
            pct = int(d["stress_contribution"] * 100)
            print(f"  {i}. {_c(d['label'], 'bold')}  ({pct}% contributing to stress)")
        print()

    # Static tip
    print(_c("  ── Immediate Advice ─────────────────────────────────", "grey"))
    print(f"  💡  {static_tip}")
    print()

    # LLM advice block
    if result.get("llm_advice"):
        print(_c("  ── Personalised AI Advice ───────────────────────────", "blue", "bold"))
        print()
        # Indent each line
        for line in result["llm_advice"].split("\n"):
            print(f"  {line}")
        print()

    # Response summary
    print(_c("  ── Your Responses ───────────────────────────────────", "grey"))
    section_map = {entry["feature"]: entry["section"] or "" for entry in QUESTIONS}
    last_section = None
    for entry in QUESTIONS:
        feat = entry["feature"]
        sec  = entry["section"] or last_section or ""
        if sec != last_section and sec:
            print(_c(f"\n  {sec}", "dim"))
            last_section = sec
        label = answer_labels.get(feat, "—")
        # Truncate long answers
        if len(label) > 60:
            label = label[:57] + "..."
        print(f"  {_c('•', 'grey')} {_c(entry['question'][:45] + '...', 'dim') if len(entry['question'])>45 else _c(entry['question'], 'dim')}")
        print(f"    {label}")

    print()
    print(_c("  Take care of yourself. You matter. 💙", "cyan", "bold"))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (standalone run)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import src.utils.config_loader as cl

    parser = argparse.ArgumentParser(description="Student Burnout Questionnaire")
    parser.add_argument("--advise", action="store_true",
                        help="Enable LLM-powered personalised advice")
    args = parser.parse_args()

    # Run questionnaire
    feature_values, answer_labels = run_questionnaire()

    # Run prediction
    cl.ConfigLoader._instance = None
    from src.models.predict_model import predict_burnout

    print(_c("\n  Calculating your burnout score...", "dim"))
    time.sleep(0.8)

    result = predict_burnout(feature_values, use_advisor=args.advise)
    display_result(result, answer_labels)
