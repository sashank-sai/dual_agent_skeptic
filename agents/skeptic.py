"""
agents/skeptic.py — The Skeptic (Cross-Examiner) node.

Responsibilities:
  - Analyse the Proponent's latest draft answer.
  - Identify the SINGLE weakest logical premise or unsupported assumption.
  - Output a strict JSON interrogation object — NO free-text fallback allowed.

JSON Schema (enforced via prompt):
  {
    "weakest_premise"          : str,   # verbatim phrase from draft
    "interrogation_type"       : str,   # factual | causal | definitional | analogical
    "question"                 : str,   # the direct Socratic challenge
    "expected_defense_criteria": [str]  # what a good defense must address
  }
"""

import os
import json
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
            temperature=0.2,        # Low temp: structured, deterministic JSON output
            max_tokens=512,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


_SKEPTIC_SYSTEM = """\
You are the Skeptic, an expert logical cross-examiner trained in Socratic method.
Your role is to identify the SINGLE weakest premise in the Proponent's answer.

You MUST respond with ONLY a valid JSON object — no preamble, no explanation, no markdown.
The JSON must follow this exact schema:

{
  "weakest_premise": "<exact phrase from the draft that is the weakest assumption>",
  "interrogation_type": "<one of: factual | causal | definitional | analogical>",
  "question": "<one precise, direct Socratic question targeting that premise>",
  "expected_defense_criteria": ["<criterion 1>", "<criterion 2>", "<criterion 3>"]
}

Rules:
- interrogation_type MUST be exactly one of: factual, causal, definitional, analogical
- question must be a single sentence ending with '?'
- expected_defense_criteria must have 2–4 items, each a concrete requirement
- Output ONLY the JSON. Nothing before or after it."""


def skeptic_node(state: AgentState) -> AgentState:
    """
    LangGraph node: analyses the Proponent's draft and emits a JSON interrogation.
    """
    llm          = _get_llm()
    draft_answer = state["draft_answer"]
    query        = state["query"]

    messages = [
        SystemMessage(content=_SKEPTIC_SYSTEM),
        HumanMessage(
            content=(
                f"Original question: {query}\n\n"
                f"Proponent's latest answer:\n{draft_answer}\n\n"
                "Identify the weakest premise and output your JSON interrogation."
            )
        ),
    ]

    response     = _llm.invoke(messages)
    raw_content  = response.content.strip()

    # ── Parse JSON with fallback ──────────────────────────────────────────
    interrogation = _parse_json_safe(raw_content)

    # Append Skeptic's turn to history
    updated_history = list(state.get("conversation_history", []))
    updated_history.append({
        "role"   : "skeptic",
        "content": json.dumps(interrogation, indent=2),
    })

    return {
        **state,
        "interrogation"        : interrogation,
        "conversation_history" : updated_history,
    }


def _parse_json_safe(raw: str) -> dict:
    """
    Attempt to parse JSON from the LLM response.
    Falls back to a structured error object if parsing fails, so the
    graph never crashes on a malformed Skeptic response.
    """
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from markdown fences or surrounding text
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Hard fallback — keeps the graph running
    return {
        "weakest_premise"          : "Unable to parse Skeptic response",
        "interrogation_type"       : "factual",
        "question"                 : "Can you clarify and expand on the key premise of your answer?",
        "expected_defense_criteria": [
            "Provide specific evidence",
            "Address logical gaps",
            "State any underlying assumptions explicitly",
        ],
        "_parse_error": True,
        "_raw_response": raw[:500],
    }
