"""
utils/logger.py — Structured turn-by-turn conversation logger.
Prints a coloured, formatted dialogue trace to stdout and can
serialise the full conversation history to a JSON-serialisable dict.
"""

import json
from datetime import datetime
from graph.state import AgentState

# ANSI colour codes for terminal output
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
MAGENTA= "\033[95m"
RED    = "\033[91m"
BLUE   = "\033[94m"
DIM    = "\033[2m"

ROLE_COLORS = {
    "proponent": CYAN,
    "skeptic"  : YELLOW,
    "arbiter"  : GREEN,
}

ROLE_LABELS = {
    "proponent": "⚡ PROPONENT",
    "skeptic"  : "🔍 SKEPTIC  ",
    "arbiter"  : "⚖️  ARBITER  ",
}


def print_header(query: str) -> None:
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}{BLUE}  DUAL-AGENT CROSS-EXAMINATION SYSTEM{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  Query:{RESET} {query}")
    print(f"{DIM}  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{BOLD}{'─'*70}{RESET}\n")


def print_turn(role: str, content: str, turn_num: int = None) -> None:
    color = ROLE_COLORS.get(role, RESET)
    label = ROLE_LABELS.get(role, role.upper())
    turn_label = f" [Turn {turn_num}]" if turn_num is not None else ""

    print(f"\n{color}{BOLD}{label}{turn_label}{RESET}")
    print(f"{color}{'─'*50}{RESET}")

    # Pretty-print JSON for Skeptic output
    if role == "skeptic":
        try:
            parsed = json.loads(content)
            print(f"  {color}Weakest Premise   :{RESET} {parsed.get('weakest_premise', '')}")
            print(f"  {color}Interrogation Type:{RESET} {parsed.get('interrogation_type', '')}")
            print(f"  {color}Question          :{RESET} {BOLD}{parsed.get('question', '')}{RESET}")
            criteria = parsed.get('expected_defense_criteria', [])
            print(f"  {color}Defense Criteria  :{RESET}")
            for c in criteria:
                print(f"    • {c}")
        except (json.JSONDecodeError, AttributeError):
            print(f"  {content}")
    else:
        # Word-wrap long lines for readability
        for line in content.split("\n"):
            print(f"  {line}")


def print_footer(final_answer: str, turn_count: int) -> None:
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}{GREEN}  ✅ FINAL ACCEPTED ANSWER  (after {turn_count} round(s)){RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")
    for line in final_answer.split("\n"):
        print(f"  {line}")
    print(f"{BOLD}{'═'*70}{RESET}\n")


def log_full_run(state: AgentState) -> dict:
    """
    Return a JSON-serialisable summary of the entire run.
    Used by the evaluation harness and Streamlit UI.
    """
    return {
        "query"               : state["query"],
        "final_answer"        : state.get("final_answer") or state.get("draft_answer"),
        "total_turns"         : state["turn_count"],
        "max_turns"           : state["max_turns"],
        "conversation_history": state.get("conversation_history", []),
        "timestamp"           : datetime.now().isoformat(),
    }


def print_run(state: AgentState, verbose: bool = True) -> None:
    """Pretty-print the entire run from a final state dict."""
    if not verbose:
        return

    print_header(state["query"])

    history = state.get("conversation_history", [])
    turn_num = 1

    for entry in history:
        role    = entry["role"]
        content = entry["content"]

        # Number proponent turns only
        tn = turn_num if role == "proponent" else None
        print_turn(role, content, tn)
        if role == "arbiter":
            turn_num += 1

    final = state.get("final_answer") or state.get("draft_answer", "")
    print_footer(final, state["turn_count"])
