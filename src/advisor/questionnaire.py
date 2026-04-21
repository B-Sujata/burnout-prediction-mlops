"""
Questionnaire Definition
All 20 human-readable questions, each with descriptive options
that silently map to the numeric values the ML model expects.

No student ever sees a number. They only see plain English descriptions.
"""

# Each question is a dict:
#   feature   → the model's internal feature name
#   question  → what the student reads
#   section   → grouped for display (Psychological / Physical / etc.)
#   options   → list of (label, numeric_value) tuples, ordered low → high stress

QUESTIONNAIRE = [

    # ── Section 1: How you've been feeling ──────────────────────────────────
    {
        "feature":  "anxiety_level",
        "section":  "How You've Been Feeling",
        "question": "How often do you feel anxious, nervous, or on edge?",
        "options": [
            ("Rarely or never — I feel calm most of the time",          0),
            ("Occasionally — maybe once or twice a week",               5),
            ("Fairly often — it comes and goes throughout the week",    10),
            ("Most of the time — it's a constant background feeling",   15),
            ("Almost constantly — it's overwhelming and hard to shake", 21),
        ],
    },
    {
        "feature":  "depression",
        "section":  "How You've Been Feeling",
        "question": "Over the past few weeks, how often have you felt\n"
                    "  down, hopeless, or like things won't get better?",
        "options": [
            ("Not at all — my mood has been generally okay",              0),
            ("A few days — brief low periods but they pass",              6),
            ("More than half the days — it lingers noticeably",          13),
            ("Nearly every day — it's hard to shake the low feeling",    20),
            ("Every day — I feel hopeless and nothing brings me joy",    27),
        ],
    },
    {
        "feature":  "self_esteem",
        "section":  "How You've Been Feeling",
        "question": "How do you generally feel about yourself and your abilities?",
        "options": [
            ("Very low — I constantly doubt myself and feel worthless",   2),
            ("Low — I often feel inadequate compared to others",          8),
            ("Mixed — some days good, some days bad",                    15),
            ("Fairly good — I believe in myself most of the time",       22),
            ("Strong — I feel confident and capable",                    28),
        ],
    },
    {
        "feature":  "mental_health_history",
        "section":  "How You've Been Feeling",
        "question": "Have you previously been diagnosed with or treated for\n"
                    "  a mental health condition (anxiety, depression, etc.)?",
        "options": [
            ("No — I have no prior mental health history",  0),
            ("Yes — I have a prior mental health history",  1),
        ],
    },

    # ── Section 2: Your body ─────────────────────────────────────────────────
    {
        "feature":  "headache",
        "section":  "Your Body",
        "question": "How often do you get headaches?",
        "options": [
            ("Never or almost never",                              0),
            ("Once a week or less",                                1),
            ("A few times a week",                                 2),
            ("Most days",                                          3),
            ("Daily — it's constant and affecting my focus",       5),
        ],
    },
    {
        "feature":  "blood_pressure",
        "section":  "Your Body",
        "question": "What is your typical blood pressure status?\n"
                    "  (If unsure, choose the option that best fits how you feel)",
        "options": [
            ("Normal — I feel fine, no issues",                    1),
            ("Slightly elevated — I've been told to watch it",     2),
            ("High — I have hypertension or feel physical strain", 3),
        ],
    },
    {
        "feature":  "breathing_problem",
        "section":  "Your Body",
        "question": "How often do you experience shortness of breath,\n"
                    "  chest tightness, or breathing difficulties?",
        "options": [
            ("Never — my breathing is always fine",                0),
            ("Rarely — only in very stressful moments",            1),
            ("Sometimes — a few times a week",                     2),
            ("Often — it happens most days",                       3),
            ("Very often — it significantly impacts my daily life", 5),
        ],
    },

    # ── Section 3: Sleep & Recovery ──────────────────────────────────────────
    {
        "feature":  "sleep_quality",
        "section":  "Sleep & Recovery",
        "question": "How would you describe your sleep quality\n"
                    "  over the past two weeks?",
        "options": [
            ("Very poor — I barely sleep or wake up constantly",   1),
            ("Poor — I often wake up tired and unrefreshed",       2),
            ("Average — some good nights, some bad",               3),
            ("Good — I sleep reasonably well most nights",         4),
            ("Excellent — I wake up feeling rested and energised", 5),
        ],
    },

    # ── Section 4: Your Environment ──────────────────────────────────────────
    {
        "feature":  "noise_level",
        "section":  "Your Environment",
        "question": "How disruptive is the noise level in your\n"
                    "  study or living space?",
        "options": [
            ("Not at all — it's quiet and easy to focus",           0),
            ("Mildly distracting — occasionally interrupts me",     1),
            ("Moderately distracting — I notice it regularly",      2),
            ("Very distracting — I struggle to concentrate",        3),
            ("Extremely distracting — I can't study or sleep",      5),
        ],
    },
    {
        "feature":  "living_conditions",
        "section":  "Your Environment",
        "question": "How comfortable and suitable are your living conditions\n"
                    "  for studying and relaxing?",
        "options": [
            ("Very poor — cramped, unhealthy, or severely lacking",  1),
            ("Poor — uncomfortable and often stressful",             2),
            ("Adequate — basic needs met but not comfortable",       3),
            ("Good — comfortable and generally supportive",          4),
            ("Excellent — comfortable, clean, and stress-free",      5),
        ],
    },
    {
        "feature":  "safety",
        "section":  "Your Environment",
        "question": "How safe do you feel in your day-to-day environment\n"
                    "  (campus, home, commute)?",
        "options": [
            ("Very unsafe — I feel threatened or at risk regularly", 1),
            ("Unsafe — I'm often worried about my safety",           2),
            ("Neutral — sometimes safe, sometimes not",              3),
            ("Safe — I feel secure most of the time",                4),
            ("Very safe — I never worry about my physical safety",   5),
        ],
    },
    {
        "feature":  "basic_needs",
        "section":  "Your Environment",
        "question": "Are your basic needs (food, housing, finances)\n"
                    "  being adequately met?",
        "options": [
            ("Not at all — I regularly go without essentials",       1),
            ("Barely — it's a constant source of stress",            2),
            ("Partially — some needs met, others are a struggle",    3),
            ("Mostly — minor financial stress but manageable",       4),
            ("Fully — I don't worry about basic needs at all",       5),
        ],
    },

    # ── Section 5: Academic Life ──────────────────────────────────────────────
    {
        "feature":  "academic_performance",
        "section":  "Academic Life",
        "question": "How would you rate your current academic performance\n"
                    "  compared to your own expectations?",
        "options": [
            ("Well below — I'm failing or significantly underperforming",  1),
            ("Below — I'm not meeting my own standards",                   2),
            ("Average — I'm getting by but not excelling",                 3),
            ("Good — I'm meeting most of my academic goals",               4),
            ("Excellent — I'm performing at or above my expectations",     5),
        ],
    },
    {
        "feature":  "study_load",
        "section":  "Academic Life",
        "question": "How heavy is your current academic workload\n"
                    "  (assignments, readings, projects)?",
        "options": [
            ("Very light — barely any work to do",                         0),
            ("Light — manageable with time to spare",                      1),
            ("Moderate — keeps me busy but it's balanced",                 2),
            ("Heavy — I'm constantly working to keep up",                  3),
            ("Overwhelming — I can't keep up no matter how hard I try",    5),
        ],
    },
    {
        "feature":  "teacher_student_relationship",
        "section":  "Academic Life",
        "question": "How would you describe your relationship with\n"
                    "  your teachers or professors?",
        "options": [
            ("Very poor — hostile, dismissive, or unsupportive",           0),
            ("Poor — they're unavailable or indifferent",                  1),
            ("Neutral — professional but no real connection",              2),
            ("Good — approachable and reasonably supportive",              3),
            ("Excellent — highly supportive and encouraging",              5),
        ],
    },
    {
        "feature":  "future_career_concerns",
        "section":  "Academic Life",
        "question": "How worried are you about your future career\n"
                    "  and life after graduation?",
        "options": [
            ("Not worried at all — I feel clear and confident",            0),
            ("Slightly worried — occasional thoughts but manageable",      1),
            ("Moderately worried — it occupies my mind regularly",         2),
            ("Very worried — it causes real stress and keeps me up",       3),
            ("Extremely worried — I feel lost and panicked about my future",5),
        ],
    },

    # ── Section 6: Social Life ────────────────────────────────────────────────
    {
        "feature":  "social_support",
        "section":  "Social Life",
        "question": "When you're struggling, how much support do\n"
                    "  you get from friends, family, or your community?",
        "options": [
            ("None — I have no one to turn to",                            0),
            ("Very little — I have one person but rarely reach out",       1),
            ("Some — I have a small support network I occasionally use",   2),
            ("Strong — I have people I can rely on when I need them",      3),
        ],
    },
    {
        "feature":  "peer_pressure",
        "section":  "Social Life",
        "question": "How much pressure do you feel from peers to perform,\n"
                    "  conform, or keep up with others?",
        "options": [
            ("None at all — I don't feel judged by my peers",              0),
            ("Mild — I notice it occasionally but it doesn't bother me",   1),
            ("Moderate — it influences my decisions sometimes",            2),
            ("Strong — I often feel I have to prove myself to others",     3),
            ("Intense — I feel constant pressure and comparison",          5),
        ],
    },
    {
        "feature":  "extracurricular_activities",
        "section":  "Social Life",
        "question": "How involved are you in activities outside of studying\n"
                    "  (sports, hobbies, clubs, exercise, socialising)?",
        "options": [
            ("Not at all — I have no time or energy for anything else",    0),
            ("Rarely — maybe once a month if I'm lucky",                   1),
            ("Sometimes — a few times a month",                            2),
            ("Regularly — a few times a week",                             3),
            ("Very active — daily involvement and I genuinely enjoy it",   5),
        ],
    },
    {
        "feature":  "bullying",
        "section":  "Social Life",
        "question": "Have you experienced bullying, harassment, or being\n"
                    "  deliberately excluded by peers or seniors?",
        "options": [
            ("Never — I've had no such experiences",                       0),
            ("Once or twice — isolated incidents that didn't persist",     1),
            ("Occasionally — it happens but infrequently",                 2),
            ("Regularly — it's an ongoing source of distress",             3),
            ("Severely — it's a major and persistent problem for me",      5),
        ],
    },
]

# Section order for display grouping
SECTION_ORDER = [
    "How You've Been Feeling",
    "Your Body",
    "Sleep & Recovery",
    "Your Environment",
    "Academic Life",
    "Social Life",
]
