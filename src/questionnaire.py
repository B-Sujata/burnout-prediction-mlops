"""
Student Burnout Questionnaire — Human-Readable CLI
===================================================
Replaces raw numeric input with descriptive multiple-choice questions.
Users never see a number. Every answer silently maps to the correct
feature value on the model's internal scale.

Run directly:
    python src/questionnaire.py
    python src/questionnaire.py --advise
    python src/questionnaire.py --no-summary
"""

import os
import sys
import argparse
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Terminal colours — degrade gracefully if not supported
try:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"
except Exception:
    GREEN = YELLOW = RED = CYAN = BOLD = DIM = RESET = ""


# ─────────────────────────────────────────────────────────────────────────────
# Question Bank
# Each entry: key, section (optional), question text, list of (label, value)
# The student sees only labels. Values go directly to the model.
# ─────────────────────────────────────────────────────────────────────────────

QUESTIONS = [
    {
        "key": "anxiety_level",
        "section": "SECTION 1 OF 5  —  Psychological Wellbeing",
        "question": "How often do you feel anxious, nervous, or on edge?",
        "options": [
            ("Rarely or never — I feel calm most of the time",              0),
            ("Occasionally — a few times a month, manageable",              5),
            ("Regularly — several times a week, noticeably affecting me",  11),
            ("Very often — almost daily, hard to shake off",               16),
            ("Constantly — overwhelming anxiety that disrupts everything", 21),
        ],
    },
    {
        "key": "self_esteem",
        "question": "How would you describe your general confidence and sense of self-worth?",
        "options": [
            ("Very confident — I genuinely believe in myself",             28),
            ("Mostly positive — occasional self-doubt but generally good", 22),
            ("Mixed — good days and bad days, quite variable",             15),
            ("Low — I often feel inadequate or not good enough",            8),
            ("Very low — I rarely feel positive about who I am",            2),
        ],
    },
    {
        "key": "mental_health_history",
        "question": "Have you ever been diagnosed with or sought professional help for a mental health condition?",
        "options": [
            ("No — no history of mental health diagnosis or treatment",     0),
            ("Yes — I have sought help or been diagnosed in the past",      1),
        ],
    },
    {
        "key": "depression",
        "question": "How often do you feel persistently sad, empty, or hopeless?",
        "options": [
            ("Rarely — I generally feel emotionally okay",                  0),
            ("Sometimes — low moods that pass within a day or two",         7),
            ("Often — I feel down several days a week",                    14),
            ("Most of the time — persistent low mood that's hard to lift", 20),
            ("Nearly every day — feeling hopeless, unable to find joy",    27),
        ],
    },
    {
        "key": "headache",
        "section": "SECTION 2 OF 5  —  Physical Health",
        "question": "How frequently do you experience headaches or migraines?",
        "options": [
            ("Almost never — headaches are very rare for me",               0),
            ("Occasionally — once or twice a month",                        1),
            ("Sometimes — once or twice a week",                            2),
            ("Frequently — several times a week",                           3),
            ("Very frequently — almost every day",                          5),
        ],
    },
    {
        "key": "blood_pressure",
        "question": "What is your general blood pressure situation?",
        "options": [
            ("Normal — within a healthy range, not a concern",              1),
            ("Slightly elevated — borderline or occasionally high",         2),
            ("High — diagnosed hypertension or consistently elevated",      3),
        ],
    },
    {
        "key": "breathing_problem",
        "question": "Do you experience difficulty breathing, chest tightness, or shortness of breath?",
        "options": [
            ("Never — no breathing difficulties at all",                    0),
            ("Rarely — only in specific situations like exercise",          1),
            ("Sometimes — occasional episodes, somewhat concerning",        2),
            ("Often — happens regularly and affects daily activities",      3),
            ("Very frequently — a persistent and disruptive problem",       5),
        ],
    },
    {
        "key": "noise_level",
        "section": "SECTION 3 OF 5  —  Living Environment",
        "question": "How noisy or disruptive is your study and living environment?",
        "options": [
            ("Very quiet — ideal environment, rarely distracted",           0),
            ("Mostly quiet — occasional noise, easy to manage",             1),
            ("Moderate — some noise that sometimes disrupts focus",         2),
            ("Noisy — frequent disruptions making it hard to concentrate",  3),
            ("Very noisy — constant disruption, nearly impossible to focus",5),
        ],
    },
    {
        "key": "living_conditions",
        "question": "How would you describe your overall living conditions (space, comfort, cleanliness)?",
        "options": [
            ("Excellent — comfortable, clean, great for studying",          5),
            ("Good — minor issues but generally fine",                      4),
            ("Okay — liveable but with noticeable problems",                3),
            ("Poor — uncomfortable or stressful living situation",          2),
            ("Very poor — seriously affecting my health or studies",        0),
        ],
    },
    {
        "key": "safety",
        "question": "How safe do you feel in your home and neighbourhood?",
        "options": [
            ("Very safe — no concerns at all",                              5),
            ("Mostly safe — occasionally cautious but generally fine",      4),
            ("Somewhat safe — some concerns I think about",                 3),
            ("Unsafe — I regularly feel at risk or threatened",             2),
            ("Very unsafe — living in fear significantly affects my life",  0),
        ],
    },
    {
        "key": "basic_needs",
        "question": "Are your basic needs being met? (food, housing security, money for essentials)",
        "options": [
            ("Fully met — no concerns about food, housing, or money",       5),
            ("Mostly met — occasional financial stress but manageable",     4),
            ("Partially — I sometimes worry about or go without basics",    3),
            ("Struggling — I regularly have difficulty meeting basic needs",2),
            ("Not met — food insecurity, housing instability, serious lack",0),
        ],
    },
    {
        "key": "academic_performance",
        "section": "SECTION 4 OF 5  —  Academic Life",
        "question": "How would you honestly rate your current academic performance?",
        "options": [
            ("Excellent — at or above my own expectations",                 5),
            ("Good — solid, mostly meeting my goals",                       4),
            ("Average — keeping up but not where I want to be",             3),
            ("Below average — struggling to pass or meet requirements",     2),
            ("Poor — significantly underperforming, at risk of failing",    0),
        ],
    },
    {
        "key": "study_load",
        "question": "How heavy is your current study workload? (assignments, exams, deadlines)",
        "options": [
            ("Very light — manageable with plenty of time to spare",        0),
            ("Light — busy but comfortable",                                1),
            ("Moderate — challenging but sustainable",                      2),
            ("Heavy — I feel overwhelmed most of the time",                 4),
            ("Extremely heavy — more than I can realistically handle",      5),
        ],
    },
    {
        "key": "teacher_student_relationship",
        "question": "How would you describe your relationship with your teachers or professors?",
        "options": [
            ("Excellent — supportive, approachable, genuinely helpful",     5),
            ("Good — mostly positive, I can seek help when needed",         4),
            ("Neutral — professional but not particularly supportive",      3),
            ("Poor — difficult to approach or communicate with",            2),
            ("Very poor — hostile, dismissive, a source of extra stress",   0),
        ],
    },
    {
        "key": "future_career_concerns",
        "question": "How worried are you about your career and future after graduating?",
        "options": [
            ("Not worried — I feel confident and optimistic",               0),
            ("Mildly concerned — some uncertainty but mostly positive",     1),
            ("Moderately worried — the future feels quite uncertain",       2),
            ("Very worried — significant fear about employment or path",    4),
            ("Extremely worried — career anxiety is consuming me daily",    5),
        ],
    },
    {
        "key": "social_support",
        "section": "SECTION 5 OF 5  —  Social Life",
        "question": "How much genuine emotional support do you have from people around you?",
        "options": [
            ("Strong — reliable people I can turn to any time",             3),
            ("Moderate — some support, though not always available",        2),
            ("Limited — I mostly handle my problems alone",                 1),
            ("None — I feel isolated with no one I can really turn to",     0),
        ],
    },
    {
        "key": "peer_pressure",
        "question": "How much pressure do you feel from peers to perform, fit in, or act a certain way?",
        "options": [
            ("None — I don't feel pressured by my peers",                   0),
            ("Mild — occasional pressure, easy to brush off",               1),
            ("Moderate — noticeable pressure that affects some choices",    3),
            ("High — peers regularly influence my decisions stressfully",   4),
            ("Intense — constant pressure that significantly affects me",   5),
        ],
    },
    {
        "key": "extracurricular_activities",
        "question": "How engaged are you in activities outside studying? (sports, clubs, hobbies, socialising)",
        "options": [
            ("Very active — regular activities I genuinely enjoy",          5),
            ("Active — participate sometimes and find it beneficial",       4),
            ("A little — some involvement but not very consistent",         3),
            ("Rarely — almost nothing outside of studying",                 1),
            ("Not at all — no time or energy for any outside activities",   0),
        ],
    },
    {
        "key": "bullying",
        "question": "Have you experienced bullying, harassment, or being treated badly by others?",
        "options": [
            ("Never — I have not experienced this",                         0),
            ("Rarely — isolated incidents, didn't significantly affect me", 1),
            ("Sometimes — occasional incidents that caused real distress",  2),
            ("Often — a recurring problem affecting me significantly",      3),
            ("Constantly — ongoing serious bullying or harassment",         5),
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_header():
    print(f"\n{BOLD}{CYAN}{'=' * 63}{RESET}")
    print(f"{BOLD}{CYAN}   Student Burnout Self-Assessment{RESET}")
    print(f"{CYAN}   Answer honestly. There are no right or wrong answers.{RESET}")
    print(f"{CYAN}   Your results are private and based on your responses.{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 63}{RESET}\n")


def print_progress(current, total):
    filled = int((current / total) * 30)
    bar = "+" * filled + "-" * (30 - filled)
    pct  = int((current / total) * 100)
    print(f"  {DIM}[{bar}] {pct}%  —  Question {current} of {total}{RESET}")


def print_section_header(text):
    print(f"\n  {BOLD}{CYAN}  {text}  {RESET}")
    print(f"  {'─' * 59}")


def ask_question(q, index, total):
    if "section" in q:
        print_section_header(q["section"])

    print()
    print_progress(index, total)
    print()
    print(f"  {BOLD}Q{index}.  {q['question']}{RESET}")
    print()

    options = q["options"]
    for i, (label, _val) in enumerate(options, 1):
        # Wrap long labels
        words = label
        print(f"  {CYAN}{i}{RESET}.  {words}")

    print()
    while True:
        try:
            raw = input(f"  {BOLD}Your choice (1-{len(options)}): {RESET}").strip()
            choice = int(raw)
            if 1 <= choice <= len(options):
                label, value = options[choice - 1]
                print(f"  {GREEN}Selected: {label}{RESET}")
                return label, value
            print(f"  {RED}Please enter a number between 1 and {len(options)}.{RESET}")
        except ValueError:
            print(f"  {RED}Please type a number, then press Enter.{RESET}")


def print_summary(answers):
    print(f"\n{'─' * 63}")
    print(f"{BOLD}  Your Answers at a Glance{RESET}")
    print(f"{'─' * 63}")
    for q in QUESTIONS:
        key = q["key"]
        label, _val = answers[key]
        short_q = (q["question"][:43] + "...") if len(q["question"]) > 46 else q["question"]
        short_l = (label[:28] + "...") if len(label) > 31 else label
        print(f"  {DIM}{short_q:<47}{RESET} {short_l}")
    print(f"{'─' * 63}\n")


def draw_bar(val, width=20):
    filled = int(val * width)
    return "+" * filled + "-" * (width - filled)


def print_result(result):
    score = result["burnout_score"]
    risk  = result["risk_level"]
    icon  = result["icon"]

    colour = GREEN if risk == "Low" else (YELLOW if risk == "Moderate" else RED)

    print(f"\n{BOLD}{colour}{'=' * 63}{RESET}")
    print(f"{BOLD}{colour}  {icon}  BURNOUT SCORE  :  {score} / 100{RESET}")
    print(f"{BOLD}{colour}      RISK LEVEL  :  {risk}{RESET}")
    print(f"{BOLD}{colour}{'=' * 63}{RESET}")
    print(f"\n  {result['advice']}\n")

    print(f"  {BOLD}Composite Index Breakdown:{RESET}")
    indices = [
        ("Emotional Strain",  result.get("emotional_strain",  0), False),
        ("Physical Stress",   result.get("physical_stress",   0), False),
        ("Academic Pressure", result.get("academic_pressure", 0), False),
        ("Social Stress",     result.get("social_stress",     0), False),
        ("Recovery Index",    result.get("recovery_index",    0), True),
    ]
    for name, val, protective in indices:
        bar   = draw_bar(val)
        note  = f"  {DIM}(higher = more protected){RESET}" if protective else ""
        print(f"  {name:<22} [{bar}]  {val:.2f}{note}")
    print()


def print_llm_advice(result):
    if not result.get("llm_advice"):
        return
    provider = result.get("llm_provider", "").title()
    model    = result.get("llm_model", "")
    print(f"\n{BOLD}{CYAN}{'─' * 63}{RESET}")
    print(f"{BOLD}{CYAN}  Personalised AI Advice  ({provider} — {model}){RESET}")
    print(f"{BOLD}{CYAN}{'─' * 63}{RESET}\n")
    for line in result["llm_advice"].splitlines():
        if line.strip():
            for wline in textwrap.wrap(line, width=60):
                print(f"  {wline}")
        else:
            print()
    if result.get("stress_drivers"):
        print(f"\n  {BOLD}Top Stress Drivers:{RESET}")
        for d in result["stress_drivers"]:
            bar = draw_bar(d["stress_contribution"], width=15)
            print(f"  * {d['label']:<35} [{bar}]  {d['stress_contribution']:.0%}")
    print(f"\n{CYAN}{'─' * 63}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main flow
# ─────────────────────────────────────────────────────────────────────────────

def run_questionnaire(
    use_advisor=False,
    show_summary=True,
    config_path="configs/config.yaml",
    advisor_config_path="configs/advisor_config.yaml",
):
    print_header()

    total   = len(QUESTIONS)
    answers = {}

    for idx, q in enumerate(QUESTIONS, 1):
        label, value  = ask_question(q, idx, total)
        answers[q["key"]] = (label, value)
        print()

    # Summary and confirmation
    if show_summary:
        print_summary(answers)
        confirm = input(
            f"  {BOLD}Press Enter to see your results  (or type 'r' to redo): {RESET}"
        ).strip().lower()
        if confirm == "r":
            print("\n  Restarting...\n")
            return run_questionnaire(use_advisor, show_summary, config_path, advisor_config_path)

    print(f"\n  {DIM}Analysing your responses...{RESET}\n")

    # Build feature dict
    feature_values = {key: val for key, (_label, val) in answers.items()}

    # Predict
    import src.utils.config_loader as cl
    cl.ConfigLoader._instance = None
    from src.models.predict_model import predict_burnout

    result = predict_burnout(
        feature_values,
        config_path=config_path,
        use_advisor=use_advisor,
        advisor_config_path=advisor_config_path,
    )

    print_result(result)
    if use_advisor:
        print_llm_advice(result)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Student Burnout Self-Assessment — Human-Friendly CLI"
    )
    parser.add_argument("--advise",      action="store_true",
                        help="Enable LLM-powered personalised advice")
    parser.add_argument("--no-summary",  action="store_true",
                        help="Skip the answer review before showing results")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--advisor-config", default="configs/advisor_config.yaml")
    args = parser.parse_args()

    try:
        run_questionnaire(
            use_advisor=args.advise,
            show_summary=not args.no_summary,
            config_path=args.config,
            advisor_config_path=args.advisor_config,
        )
    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Assessment cancelled. Take care!{RESET}\n")
        sys.exit(0)
