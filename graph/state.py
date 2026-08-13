"""
graph/state.py — Shared state schema for the LangGraph StateGraph.
All agent nodes read from and write to this TypedDict.
"""

from typing import TypedDict, List, Optional


class ConversationTurn(TypedDict):
    role: str       # "proponent" | "skeptic" | "arbiter"
    content: str    # raw text content of that turn


class AgentState(TypedDict):
    """
    The single source of truth passed between all graph nodes.

    Fields
    ------
    query               : The original research question.
    draft_answer        : Proponent's current best answer (updated each turn).
    interrogation       : Skeptic's latest JSON challenge (dict or None).
    turn_count          : Number of completed Skeptic→Proponent cycles.
    max_turns           : Hard ceiling on cycles (from config).
    conversation_history: Ordered list of all agent turns for context injection.
    final_answer        : Set by Arbiter when the loop is accepted/terminated.
    verdict             : "continue" → loop again | "accept" → end.
    """
    query               : str
    draft_answer        : str
    interrogation       : Optional[dict]
    turn_count          : int
    max_turns           : int
    conversation_history: List[ConversationTurn]
    final_answer        : Optional[str]
    verdict             : Optional[str]   # "continue" | "accept"
