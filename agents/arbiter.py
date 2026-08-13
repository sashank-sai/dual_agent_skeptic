"""
agents/arbiter.py — The Arbiter (Router) node.

Responsibilities:
  - After each Proponent defense, decide: "continue" or "accept".
  - Decision logic (in priority order):
      1. If turn_count >= max_turns → force accept (loop termination guard).
      2. Use a lightweight LLM call to score the defense quality against
         the Skeptic's expected_defense_criteria.
      3. If defense meets criteria → accept. Else → continue.
  - Sets `verdict`, `turn_count`, and `final_answer` in state.
"""

import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from graph.state import AgentState

load_dotenv()

_llm = None

def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=config.MODEL_NAME,
            temperature=0.1,    # Near-deterministic for routing decisions
            max_tokens=128,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


_ARBITER_SYSTEM = """\
You are the Arbiter, a neutral evaluator of logical sufficiency.
You will be shown:
  1. A Skeptic's interrogation criteria (what a good defense must address).
  2. The Proponent's defense response.

Your only job: decide if the defense adequately addresses ALL the criteria.

Respond with EXACTLY one of these two tokens:
  ACCEPT   — if the defense satisfactorily addresses the criteria
  CONTINUE — if the defense is incomplete, evasive, or misses a criterion

Output ONLY "ACCEPT" or "CONTINUE". Nothing else."""


def arbiter_node(state: AgentState) -> AgentState:
    """
    LangGraph node: evaluates the Proponent's latest defense and sets verdict.
    Increments turn_count after each evaluation.
    """
    turn_count = state["turn_count"]
    max_turns  = state["max_turns"]
    new_count  = turn_count + 1

    # ── Hard termination guard ────────────────────────────────────────────
    if new_count >= max_turns:
        updated_history = list(state.get("conversation_history", []))
        updated_history.append({
            "role"   : "arbiter",
            "content": f"[Turn limit {max_turns} reached] Force-accepting current answer.",
        })
        return {
            **state,
            "turn_count"          : new_count,
            "verdict"             : "accept",
            "final_answer"        : state["draft_answer"],
            "conversation_history": updated_history,
        }

    # ── LLM-based defense quality evaluation ─────────────────────────────
    interrogation = state.get("interrogation", {})
    criteria      = interrogation.get("expected_defense_criteria", [])
    criteria_text = "\n".join(f"  - {c}" for c in criteria)
    draft_answer  = state["draft_answer"]

    messages = [
        SystemMessage(content=_ARBITER_SYSTEM),
        HumanMessage(
            content=(
                f"Skeptic's expected defense criteria:\n{criteria_text}\n\n"
                f"Proponent's defense:\n{draft_answer}\n\n"
                "Does this defense adequately address ALL the criteria? "
                "Output ACCEPT or CONTINUE."
            )
        ),
    ]

    response = _get_llm().invoke(messages)
    raw      = response.content.strip().upper()

    # Robustly extract verdict even if model adds surrounding words
    if "ACCEPT" in raw:
        verdict = "accept"
    else:
        verdict = "continue"

    # Build Arbiter's history entry
    updated_history = list(state.get("conversation_history", []))
    updated_history.append({
        "role"   : "arbiter",
        "content": f"Verdict: {verdict.upper()} (Turn {new_count}/{max_turns})",
    })

    final_answer = state["draft_answer"] if verdict == "accept" else None

    return {
        **state,
        "turn_count"          : new_count,
        "verdict"             : verdict,
        "final_answer"        : final_answer,
        "conversation_history": updated_history,
    }
