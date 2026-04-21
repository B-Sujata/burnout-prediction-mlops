"""
Prompt Templates for the LLM Burnout Advisor.
Uses LangChain's ChatPromptTemplate for structured multi-turn prompts.
"""

def _get_prompt_classes():
    from langchain_core.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    return ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — sets the LLM's persona
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """{persona}

When giving advice, always structure your response in these exact sections:
1. **Understanding Your Situation** — A 2-3 sentence empathetic summary of what the data reveals about this student.
2. **Your Top Stress Drivers** — Explain the {n_drivers} biggest factors contributing to their burnout, in plain language.
3. **Immediate Actions (This Week)** — 3 specific, practical steps they can take right now.
4. **Short-Term Goals (Next Month)** — 3 strategies to address the root causes over the coming weeks.
5. **When to Seek Help** — Clear, non-stigmatising guidance on whether professional support is recommended.

Keep your tone warm, supportive, and specific to this student's data. Never be generic.
"""

HUMAN_TEMPLATE = """Here is the burnout assessment for a student. Please provide personalised advice.

**BURNOUT ASSESSMENT**
- Burnout Score: {burnout_score}/100
- Risk Level: {risk_level}

**STUDENT PROFILE (Feature Values)**
{feature_summary}

**TOP STRESS DRIVERS IDENTIFIED**
{stress_drivers}

**COMPOSITE INDEX BREAKDOWN**
- Emotional Strain Index: {emotional_strain:.2f}/1.0
- Physical Stress Index: {physical_stress:.2f}/1.0
- Academic Pressure Index: {academic_pressure:.2f}/1.0
- Social Stress Index: {social_stress:.2f}/1.0
- Recovery Index: {recovery_index:.2f}/1.0 (higher = better protected)

Please provide a detailed, personalised advice plan for this student.
"""


def build_advisor_prompt(persona: str, n_drivers: int):
    """
    Build the full ChatPromptTemplate for the burnout advisor.

    Args:
        persona: The system persona string from advisor_config.yaml
        n_drivers: Number of top stress drivers to surface

    Returns:
        LangChain ChatPromptTemplate ready for use in a chain
    """
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate = _get_prompt_classes()

    system_msg = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
    human_msg  = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)

    return ChatPromptTemplate.from_messages([system_msg, human_msg]).partial(
        persona=persona,
        n_drivers=n_drivers,
    )


FOLLOWUP_TEMPLATE = """The student has a follow-up question about their burnout advice:

{followup_question}

Please answer in the context of their burnout score of {burnout_score}/100 ({risk_level} risk).
Keep your answer concise (3-5 sentences) and actionable.
"""

def get_followup_prompt():
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate = _get_prompt_classes()
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{persona}"),
        HumanMessagePromptTemplate.from_template(FOLLOWUP_TEMPLATE),
    ])
