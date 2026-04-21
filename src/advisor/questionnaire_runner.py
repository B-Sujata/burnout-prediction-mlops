"""
CLI Questionnaire Runner
Presents the 20-question burnout assessment to students in plain English.
No numbers shown — only descriptive multiple-choice options.
Feeds answers into the ML model and optionally the LLM advisor.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.advisor.questionnaire import QUESTIONNAIRE, SECTION_ORDER


# ─────────────────────────────────────────────────────────────────────────────
# Terminal colours (works on Mac, Linux, Windows 10+)
# ─────────────────────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"

def bold(s):   return f"{C.BOLD}{s}{C.RESET}"
def cyan(s):   return f"{C.CYAN}{s}{C.RESET}"
def green(s):  return f"{C.GREEN}{s}{C.RESET}"
def yellow(s): return f"{C.YELLOW}{s}{C.RESET}"
def red(s):    return f"{C.RED}{s}{C.RESET}"
def dim(s):    return f"{C.DIM}{s}{C.RESET}"
def blue(s):   return f"{C.BLUE}{s}{C.RESET}"
def magenta(s):return f"{C.MAGENTA}{s}{C.RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def clear_line():
    print("\033[2K\033[1A", end="", flush=True)

def progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / total)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = int(100 * current / total)
    return f"{cyan(bar)} {bold(str(pct)+'%')}"

def print_header():
    print()
    print(bold(cyan("╔══════════════════════════════════════════════════════════╗")))
    print(bold(cyan("║      🧠  Student Burnout Self-Assessment                 ║")))
    print(bold(cyan("║      Answer honestly — there are no right or wrong       ║")))
    print(bold(cyan("║      answers. This takes about 3–4 minutes.              ║")))
    print(bold(cyan("╚══════════════════════════════════════════════════════════╝")))
    print()
    print(dim("  Your answers are used only to calculate your burnout score."))
    print(dim("  No data is stored or shared."))
    print()

def print_section_header(section: str, is_new: bool):
    if is_new:
        print()
        print(f"  {magenta('─' * 55)}")
        print(f"  {bold(magenta('  📋  ' + section.upper()))}")
        print(f"  {magenta('─' * 55)}")
        print()

def print_result(score: float, risk: str, icon: str, advice: str):
    print()
    print(bold(cyan("╔══════════════════════════════════════════════════════════╗")))
    print(bold(cyan("║                  YOUR BURNOUT ASSESSMENT                 ║")))
    print(bold(cyan("╚══════════════════════════════════════════════════════════╝")))
    print()

    # Score bar
    bar_width = 40
    filled    = int(bar_width * score / 100)
    if score <= 33:   bar_color = green
    elif score <= 66: bar_color = yellow
    else:             bar_color = red

    bar = bar_color("█" * filled) + dim("░" * (bar_width - filled))
    print(f"  Score   {bar}  {bold(str(score))}/100")
    print()

    # Risk level
    risk_display = {
        "Low":      green(f"  🟢  LOW RISK          — You're doing well"),
        "Moderate": yellow(f"  🟡  MODERATE RISK     — Worth paying attention to"),
        "High":     red(f"  🔴  HIGH RISK         — Please seek support soon"),
    }
    print(f"  {bold('Risk Level')}  {risk_display.get(risk, risk)}")
    print()

    # Divider
    print(f"  {dim('─' * 57)}")
    print()
    print(f"  {bold('💡 What this means for you:')}")
    print()
    # Word-wrap advice at 55 chars
    words = advice.split()
    line = "     "
    for word in words:
        if len(line) + len(word) + 1 > 60:
            print(line)
            line = "     " + word
        else:
            line += (" " if line.strip() else "") + word
    if line.strip():
        print(line)
    print()


def print_stress_drivers(drivers: list):
    print(f"  {dim('─' * 57)}")
    print()
    print(f"  {bold('📊 Your top stress contributors:')}")
    print()
    icons = ["🔴", "🟠", "🟡"]
    for i, d in enumerate(drivers[:3]):
        pct  = int(d["stress_contribution"] * 100)
        icon = icons[i] if i < len(icons) else "•"
        bar  = "▓" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"  {icon}  {bold(d['label'])}")
        print(f"      {red(bar) if pct > 60 else yellow(bar) if pct > 40 else green(bar)}  {pct}% stress contribution")
        print()


def print_llm_advice(advice_text: str, provider: str, model: str):
    print(f"  {dim('─' * 57)}")
    print()
    print(f"  {bold('🤖 Personalised AI Advice')}  {dim(f'(via {provider} / {model})')}")
    print()
    # Print advice with indent, preserving markdown-style bold (**text**)
    for line in advice_text.split("\n"):
        # Simple bold: replace **text** with terminal bold
        import re
        line = re.sub(r'\*\*(.+?)\*\*', lambda m: bold(m.group(1)), line)
        if line.strip():
            print(f"  {line}")
        else:
            print()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Core questionnaire loop
# ─────────────────────────────────────────────────────────────────────────────

def ask_question(
    q: dict,
    q_num: int,
    total: int,
    current_section: str,
) -> tuple[float, str]:
    """
    Display a single question and collect the user's answer.
    Returns (numeric_value, section_name).
    """
    section = q["section"]
    is_new_section = section != current_section

    print_section_header(section, is_new_section)

    # Progress
    print(f"  {progress_bar(q_num - 1, total)}  "
          f"{dim(f'Question {q_num} of {total}')}")
    print()

    # Question text (supports multi-line via \n in the string)
    q_lines = q["question"].split("\n")
    print(f"  {bold(q_lines[0].strip())}")
    for extra in q_lines[1:]:
        print(f"  {extra.strip()}")
    print()

    # Options
    options = q["options"]
    for i, (label, _value) in enumerate(options, 1):
        num = cyan(f"  {i}.")
        print(f"{num}  {label}")
    print()

    # Input
    while True:
        try:
            raw = input(f"  {bold('Your answer')} {dim('[1–' + str(len(options)) + ']')}: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                chosen_label, chosen_value = options[idx]
                print(f"  {green('✓')}  {dim(chosen_label)}")
                print()
                return float(chosen_value), section
            else:
                print(f"  {red('⚠')}  Please enter a number between 1 and {len(options)}")
        except (ValueError, EOFError):
            print(f"  {red('⚠')}  Please enter a number between 1 and {len(options)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_questionnaire(
    use_advisor: bool = False,
    config_path: str = "configs/config.yaml",
    advisor_config_path: str = "configs/advisor_config.yaml",
) -> dict:
    """
    Run the full 20-question assessment and return the prediction result.

    Args:
        use_advisor:          Whether to invoke the LLM advisor after prediction
        config_path:          Path to ML pipeline config
        advisor_config_path:  Path to advisor config

    Returns:
        Full prediction result dict from predict_burnout()
    """
    print_header()
    input(f"  {bold('Press Enter to begin the assessment...')}")
    print()

    answers       = {}
    current_sect  = ""
    total_q       = len(QUESTIONNAIRE)

    for i, q in enumerate(QUESTIONNAIRE, 1):
        value, current_sect = ask_question(q, i, total_q, current_sect)
        answers[q["feature"]] = value

    # ── Run prediction ────────────────────────────────────────────────────
    print()
    print(f"  {dim('Calculating your burnout score...')}", end="", flush=True)
    time.sleep(0.6)   # brief pause so it doesn't feel instant
    print(f"\r  {green('✓')} Assessment complete!                    ")
    print()

    # Import here to keep questionnaire module lightweight
    import src.utils.config_loader as cl
    cl.ConfigLoader._instance = None
    from src.models.predict_model import predict_burnout

    result = predict_burnout(
        answers,
        config_path=config_path,
        use_advisor=use_advisor,
        advisor_config_path=advisor_config_path,
    )

    # ── Display results ───────────────────────────────────────────────────
    print_result(
        score  = result["burnout_score"],
        risk   = result["risk_level"],
        icon   = result["icon"],
        advice = result["advice"],
    )

    # Stress drivers (always shown — no API key needed)
    if result.get("stress_drivers"):
        print_stress_drivers(result["stress_drivers"])

    # LLM advice
    if use_advisor and result.get("llm_advice"):
        print_llm_advice(
            advice_text = result["llm_advice"],
            provider    = result.get("llm_provider", "unknown"),
            model       = result.get("llm_model", "unknown"),
        )
    elif use_advisor and not result.get("llm_advice"):
        print(f"  {yellow('⚠')}  {dim('LLM advice unavailable — set ANTHROPIC_API_KEY or OPENAI_API_KEY')}")
        print()

    print(bold(cyan("╔══════════════════════════════════════════════════════════╗")))
    print(bold(cyan("║  Thank you for completing the assessment.                ║")))
    print(bold(cyan("║  If you are struggling, please speak to someone you      ║")))
    print(bold(cyan("║  trust or contact your university counselling service.   ║")))
    print(bold(cyan("╚══════════════════════════════════════════════════════════╝")))
    print()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Student Burnout Self-Assessment (Human-Friendly CLI)"
    )
    parser.add_argument(
        "--advise", action="store_true",
        help="Enable LLM-powered personalised advice after the assessment "
             "(requires ANTHROPIC_API_KEY or OPENAI_API_KEY)"
    )
    args = parser.parse_args()

    run_questionnaire(use_advisor=args.advise)