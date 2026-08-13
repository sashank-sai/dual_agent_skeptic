"""
agents/proponent.py — The Proponent (Actor) node.

Responsibilities:
  - Turn 0 : Generate an initial structured answer (Premise → Reasoning → Conclusion).
  - Turn N>0: Receive the Skeptic's JSON interrogation and either DEFEND or REVISE
              the challenged premise, providing new evidence or logic.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from graph.state import AgentState

load_dotenv()

# ── LLM client (shared singleton) ────────────────────────────────────────────
_llm = None

def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


# ── System prompts ────────────────────────────────────────────────────────────
_INITIAL_SYSTEM = """\
You are the Proponent, a rigorous analytical agent.
Your task is to answer a research question with a well-structured response.

Always format your answer in exactly three labelled sections:
  PREMISE     : State the key factual assumption your answer rests on.
  REASONING   : Explain the logical chain from premise to conclusion.
  CONCLUSION  : State your final, direct answer to the question.

Be precise. Avoid hedging. Do not use bullet points inside sections."""

_DEFENSE_SYSTEM = """\
You are the Proponent, defending your previous answer under Socratic cross-examination.
A Skeptic has challenged one of your premises. You must respond critically.

Your response MUST follow this format:
  DEFENSE/REVISION : State whether you are DEFENDING or REVISING the challenged premise.
  JUSTIFICATION    : Provide specific evidence, logic, or clarification.
  UPDATED CONCLUSION: Restate your answer in light of this justification.

Do not be sycophantic. If your original premise was correct, defend it strongly."""


# ── Node function ─────────────────────────────────────────────────────────────
def proponent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: generates or refines the Proponent's answer.
    Returns a partial state dict with updated fields.
    """
    llm = _get_llm()
    query = state["query"]
    turn  = state["turn_count"]

    if turn == 0:
        # ── First answer: no prior context ────────────────────────────────
        messages = [
            SystemMessage(content=_INITIAL_SYSTEM),
            HumanMessage(content=f"Question: {query}"),
        ]
    else:
        # ── Defense turn: inject interrogation and conversation history ───
        interrogation = state.get("interrogation", {})
        history_text  = _format_history(state["conversation_history"])

        challenge_text = (
            f"Weakest premise identified: \"{interrogation.get('weakest_premise', '')}\"\n"
            f"Interrogation type        : {interrogation.get('interrogation_type', '')}\n"
            f"Skeptic's question        : {interrogation.get('question', '')}\n"
            f"Expected defense criteria : {interrogation.get('expected_defense_criteria', [])}"
        )

        messages = [
            SystemMessage(content=_DEFENSE_SYSTEM),
            HumanMessage(
                content=(
                    f"Original question: {query}\n\n"
                    f"--- Conversation so far ---\n{history_text}\n\n"
                    f"--- Skeptic's Challenge ---\n{challenge_text}\n\n"
                    "Provide your defense or revision now."
                )
            ),
        ]

    response     = llm.invoke(messages)
    draft_answer = response.content.strip()

    # Append this turn to history
    updated_history = list(state.get("conversation_history", []))
    updated_history.append({"role": "proponent", "content": draft_answer})

    return {
        **state,
        "draft_answer"        : draft_answer,
        "conversation_history": updated_history,
    }


def _format_history(history: list) -> str:
    """Convert conversation history list to a readable string for injection."""
    lines = []
    for turn in history:
        role    = turn["role"].upper()
        content = turn["content"]
        lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines) if lines else "(none)"
