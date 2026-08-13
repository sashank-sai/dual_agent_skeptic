"""
graph/workflow.py — Assembles the LangGraph StateGraph.

Flow:
    START
      │
      ▼
  [proponent]  ←──────────────────────────┐
      │                                   │
      ▼                                   │ verdict == "continue"
  [skeptic]                               │
      │                                   │
      ▼                                   │
  [arbiter] ──── verdict == "accept" ──► END
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from agents.proponent import proponent_node
from agents.skeptic    import skeptic_node
from agents.arbiter    import arbiter_node
import config


def _route_after_arbiter(state: AgentState) -> str:
    """Conditional edge: read the Arbiter's verdict and pick next node."""
    return "proponent" if state.get("verdict") == "continue" else END


def build_graph():
    """
    Construct and compile the dual-agent StateGraph.
    Returns a compiled LangGraph application ready for .invoke() or .stream().
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("proponent", proponent_node)
    builder.add_node("skeptic",   skeptic_node)
    builder.add_node("arbiter",   arbiter_node)

    # ── Entry point ───────────────────────────────────────────────────────
    builder.set_entry_point("proponent")

    # ── Edges ─────────────────────────────────────────────────────────────
    builder.add_edge("proponent", "skeptic")
    builder.add_edge("skeptic",   "arbiter")

    builder.add_conditional_edges(
        "arbiter",
        _route_after_arbiter,
        {
            "proponent": "proponent",   # continue loop
            END        : END,           # accept → terminate
        },
    )

    return builder.compile()


def run_query(query: str, max_turns: int = None) -> AgentState:
    """
    High-level helper: build graph, set initial state, invoke, return final state.

    Parameters
    ----------
    query     : The question to research.
    max_turns : Override config.MAX_TURNS if provided.
    """
    if max_turns is None:
        max_turns = config.MAX_TURNS

    graph = build_graph()

    initial_state: AgentState = {
        "query"               : query,
        "draft_answer"        : "",
        "interrogation"       : None,
        "turn_count"          : 0,
        "max_turns"           : max_turns,
        "conversation_history": [],
        "final_answer"        : None,
        "verdict"             : None,
    }

    final_state = graph.invoke(initial_state)
    return final_state
