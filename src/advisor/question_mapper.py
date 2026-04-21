"""
Question Mapper
===============
Translates human-friendly descriptive answers → numeric feature values
that the ML model understands.

A student never sees a number like "Anxiety Level = 15".
They pick from options like "Almost always — it significantly affects my day."
We map that to 17 internally.

Each question has:
  - question    : What the student reads
  - context     : Optional 1-line helper (shown in grey)
  - options     : List of (display_label, numeric_value) pairs
  - feature     : Which model feature this maps to
"""

QUESTIONS = [

    # ── Psychological ─────────────────────────────────────────────────────────

    {
        "feature":  "anxiety_level",
        "section":  "Psychological Wellbeing",
        "question": "How often do you feel anxious, nervous, or overwhelmed?",
        "context":  "Think about the past 2 weeks.",
        "options": [
            ("Never or very rarely",                                0),
            ("Occasionally — a few times",                         4),
            ("Sometimes — it comes and goes",                      8),
            ("Often — most days I feel this way",                 13),
            ("Almost always — it significantly affects my day",   17),
            ("Constantly — I feel anxious nearly all the time",   21),
        ],
    },

    {
        "feature":  "self_esteem",
        "section":  "Psychological Wellbeing",
        "question": "How confident and positive do you feel about yourself overall?",
        "context":  "Consider how you see your own worth and abilities.",
        "options": [
            ("Very low — I often feel worthless or incapable",     3),
            ("Low — I doubt myself a lot",                        10),
            ("Moderate — some good days, some bad",               17),
            ("Good — I generally feel capable and valued",        24),
            ("Very high — I feel confident and self-assured",     30),
        ],
    },

    {
        "feature":  "mental_health_history",
        "section":  "Psychological Wellbeing",
        "question": "Have you ever been diagnosed with or treated for a mental health condition?",
        "context":  "This includes depression, anxiety disorders, ADHD, etc. All answers are private.",
        "options": [
            ("No",  0),
            ("Yes", 1),
        ],
    },

    {
        "feature":  "depression",
        "section":  "Psychological Wellbeing",
        "question": "In the past 2 weeks, how often have you felt down, hopeless, or lost interest in things you usually enjoy?",
        "context":  "Be honest — there are no wrong answers.",
        "options": [
            ("Not at all",                                         0),
            ("A few days",                                         5),
            ("About half the days",                               13),
            ("More than half the days",                           19),
            ("Nearly every day",                                  25),
            ("Every day — it's severely impacting my life",       27),
        ],
    },

    # ── Physical ──────────────────────────────────────────────────────────────

    {
        "feature":  "headache",
        "section":  "Physical Health",
        "question": "How often do you experience headaches?",
        "context":  "Include tension headaches and migraines.",
        "options": [
            ("Never",                              0),
            ("Rarely — once a month or less",      1),
            ("Sometimes — a few times a month",    2),
            ("Often — weekly",                     3),
            ("Very often — several times a week",  4),
            ("Daily or almost daily",              5),
        ],
    },

    {
        "feature":  "blood_pressure",
        "section":  "Physical Health",
        "question": "What is your blood pressure status?",
        "context":  "If unsure, select 'Normal / I don't know'.",
        "options": [
            ("Normal / I don't know",   1),
            ("Slightly elevated",        2),
            ("High — I've been told by a doctor", 3),
        ],
    },

    {
        "feature":  "sleep_quality",
        "section":  "Physical Health",
        "question": "How would you rate the quality of your sleep?",
        "context":  "Think about how rested you feel in the morning.",
        "options": [
            ("Very poor — I barely sleep or feel completely unrested", 0),
            ("Poor — I wake up often or feel tired most mornings",     1),
            ("Fair — it's okay but not great",                         2),
            ("Good — I usually feel reasonably rested",                3),
            ("Very good — I sleep well and wake up refreshed",         4),
            ("Excellent — my sleep is consistently great",             5),
        ],
    },

    {
        "feature":  "breathing_problem",
        "section":  "Physical Health",
        "question": "How often do you experience shortness of breath or difficulty breathing?",
        "context":  "Include stress-related breathing difficulties.",
        "options": [
            ("Never",                             0),
            ("Rarely",                            1),
            ("Sometimes",                         2),
            ("Often",                             3),
            ("Very often — it worries me",        4),
            ("Constantly — it's a serious issue", 5),
        ],
    },

    # ── Environmental ─────────────────────────────────────────────────────────

    {
        "feature":  "noise_level",
        "section":  "Your Environment",
        "question": "How noisy or disruptive is your study and living environment?",
        "context":  "Think about where you sleep, study, and spend most time.",
        "options": [
            ("Very quiet — ideal for focus and rest",                  0),
            ("Mostly quiet — occasional disturbances",                 1),
            ("Moderate — some noise but manageable",                   2),
            ("Quite noisy — hard to concentrate sometimes",            3),
            ("Very noisy — it seriously disrupts my study and sleep",  5),
        ],
    },

    {
        "feature":  "living_conditions",
        "section":  "Your Environment",
        "question": "How would you describe your living conditions (accommodation, cleanliness, space)?",
        "context":  "Home, hostel, PG, or wherever you primarily live.",
        "options": [
            ("Very poor — cramped, dirty, or unsafe",  0),
            ("Poor — uncomfortable most of the time",  1),
            ("Fair — basic but manageable",             2),
            ("Good — comfortable and adequate",         4),
            ("Excellent — very comfortable and clean",  5),
        ],
    },

    {
        "feature":  "safety",
        "section":  "Your Environment",
        "question": "How safe do you feel in your college/home environment?",
        "context":  "Consider physical safety and sense of security.",
        "options": [
            ("Very unsafe — I feel threatened",        0),
            ("Somewhat unsafe — I worry often",        1),
            ("Neutral — neither safe nor unsafe",      2),
            ("Safe — I feel mostly secure",            4),
            ("Very safe — I feel completely secure",   5),
        ],
    },

    {
        "feature":  "basic_needs",
        "section":  "Your Environment",
        "question": "How well are your basic needs met — food, water, shelter, and clothing?",
        "context":  "Financial stress about essentials counts here.",
        "options": [
            ("Not at all — I struggle with basic necessities",   0),
            ("Poorly — I often go without things I need",        1),
            ("Somewhat — it's a constant source of stress",      2),
            ("Mostly — minor issues occasionally",               4),
            ("Fully — no concerns in this area",                 5),
        ],
    },

    # ── Academic ──────────────────────────────────────────────────────────────

    {
        "feature":  "academic_performance",
        "section":  "Academic Life",
        "question": "How would you rate your current academic performance?",
        "context":  "Relative to your own expectations, not a class ranking.",
        "options": [
            ("Very poor — I'm struggling significantly",       0),
            ("Poor — below where I want to be",               1),
            ("Average — passing but not thriving",             2),
            ("Good — I'm doing well",                         4),
            ("Excellent — performing at or above my best",    5),
        ],
    },

    {
        "feature":  "study_load",
        "section":  "Academic Life",
        "question": "How heavy is your current study or coursework load?",
        "context":  "Consider assignments, exams, projects, and self-study hours.",
        "options": [
            ("Very light — minimal work",                              0),
            ("Light — manageable without much stress",                 1),
            ("Moderate — busy but balanced",                           2),
            ("Heavy — I often feel behind or overwhelmed",             4),
            ("Extremely heavy — I can barely keep up",                 5),
        ],
    },

    {
        "feature":  "teacher_student_relationship",
        "section":  "Academic Life",
        "question": "How would you describe your relationship with your teachers or professors?",
        "context":  "Think about approachability, support, and fairness.",
        "options": [
            ("Very poor — they are unapproachable or unfair",          0),
            ("Poor — I rarely feel supported",                         1),
            ("Average — neutral, professional but distant",            2),
            ("Good — they are generally helpful and approachable",     4),
            ("Excellent — very supportive and encouraging",            5),
        ],
    },

    {
        "feature":  "future_career_concerns",
        "section":  "Academic Life",
        "question": "How worried are you about your future career or job prospects?",
        "context":  "Think about placements, further studies, or career path uncertainty.",
        "options": [
            ("Not worried at all — I feel clear and confident",    0),
            ("Slightly worried — a few questions but manageable",  1),
            ("Moderately worried — it occupies my mind sometimes", 2),
            ("Very worried — it stresses me out regularly",        4),
            ("Extremely anxious — it's a major source of distress",5),
        ],
    },

    # ── Social ────────────────────────────────────────────────────────────────

    {
        "feature":  "social_support",
        "section":  "Social Life",
        "question": "How much emotional support do you receive from friends, family, or others you trust?",
        "context":  "Think about who you could talk to if things got really difficult.",
        "options": [
            ("None — I feel completely alone",                          0),
            ("Very little — I have people but rarely feel supported",   1),
            ("Moderate — I have some support when I really need it",    2),
            ("Strong — I feel genuinely supported by people close to me",3),
        ],
    },

    {
        "feature":  "peer_pressure",
        "section":  "Social Life",
        "question": "How much pressure do you feel from classmates or friends to perform, conform, or keep up?",
        "context":  "Academic competition, social media comparison, lifestyle pressure.",
        "options": [
            ("None — I don't feel pressured by peers at all",         0),
            ("Very little — barely noticeable",                       1),
            ("Some — it affects me occasionally",                     2),
            ("Quite a bit — it's a regular source of stress",         4),
            ("Extreme — peer pressure strongly affects my wellbeing", 5),
        ],
    },

    {
        "feature":  "extracurricular_activities",
        "section":  "Social Life",
        "question": "How often do you engage in hobbies, sports, clubs, or activities outside of studying?",
        "context":  "These are recovery activities — anything that recharges you.",
        "options": [
            ("Never — I have no time or interest",                          0),
            ("Rarely — maybe once a month or less",                         1),
            ("Sometimes — a few times a month",                             2),
            ("Often — at least weekly",                                     4),
            ("Very often — it's a regular and important part of my life",   5),
        ],
    },

    {
        "feature":  "bullying",
        "section":  "Social Life",
        "question": "Have you experienced bullying, harassment, or being made to feel inferior by others?",
        "context":  "This includes online, verbal, and social exclusion. All answers are confidential.",
        "options": [
            ("Never",                                                          0),
            ("Rarely — it happened once or twice",                            1),
            ("Sometimes — it happens occasionally",                           2),
            ("Often — it happens regularly and bothers me",                   4),
            ("Severely — it is a major source of pain and distress for me",   5),
        ],
    },
]


# ── Grouping for display ──────────────────────────────────────────────────────

SECTIONS = [
    "Psychological Wellbeing",
    "Physical Health",
    "Your Environment",
    "Academic Life",
    "Social Life",
]


def get_questions_by_section() -> dict:
    """Returns questions grouped by section for UI rendering."""
    grouped = {s: [] for s in SECTIONS}
    for i, q in enumerate(QUESTIONS):
        grouped[q["section"]].append({**q, "index": i})
    return grouped


def run_friendly_questionnaire() -> dict:
    """
    Runs an interactive CLI questionnaire using descriptive option labels.
    Returns a dict of {feature_name: numeric_value} ready for predict_burnout().
    """
    print("\n" + "═" * 65)
    print("  🧠  Student Burnout Assessment")
    print("  Answer honestly — there are no right or wrong answers.")
    print("  All responses are private and used only for your assessment.")
    print("═" * 65)

    answers = {}
    current_section = None

    for q in QUESTIONS:
        # Print section header when it changes
        if q["section"] != current_section:
            current_section = q["section"]
            print(f"\n  ── {current_section} " + "─" * (45 - len(current_section)))

        print(f"\n  {q['question']}")
        if q.get("context"):
            print(f"  ({q['context']})")
        print()

        for i, (label, _) in enumerate(q["options"], 1):
            print(f"    {i}. {label}")

        while True:
            try:
                raw = input(f"\n  Your answer [1–{len(q['options'])}]: ").strip()
                choice = int(raw)
                if 1 <= choice <= len(q["options"]):
                    selected_label, numeric_val = q["options"][choice - 1]
                    answers[q["feature"]] = numeric_val
                    print(f"  ✓ Noted: \"{selected_label}\"")
                    break
                else:
                    print(f"  ⚠  Please enter a number between 1 and {len(q['options'])}")
            except ValueError:
                print(f"  ⚠  Please enter a number between 1 and {len(q['options'])}")

    print("\n" + "═" * 65)
    print("  ✅  All questions answered. Calculating your burnout score...")
    print("═" * 65 + "\n")
    return answers


def map_answers_to_features(answers: dict) -> dict:
    """
    Validates that all 20 features are present in the answers dict.
    Fills any missing features with their midpoint value (failsafe).
    """
    all_features = {q["feature"]: q for q in QUESTIONS}
    result = {}

    for feature, q_data in all_features.items():
        if feature in answers:
            result[feature] = answers[feature]
        else:
            # Fallback: midpoint of options
            vals = [v for _, v in q_data["options"]]
            result[feature] = vals[len(vals) // 2]

    return result
