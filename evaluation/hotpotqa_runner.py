"""
evaluation/hotpotqa_runner.py — HotpotQA benchmark harness.

Evaluates the dual-agent pipeline on N questions from HotpotQA
(distractor split, validation set) and compares against a
single-agent baseline (Proponent-only, no cross-examination).

Metrics: Exact Match (EM) and Token-Level F1
Results: saved to results/hotpotqa_results.json
"""

import os
import sys
import json
import re
import string
import time
from collections import Counter
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from graph.workflow import run_query, build_graph
from graph.state import AgentState
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import config

load_dotenv()

RESULTS_DIR = config.RESULTS_DIR


# ── Metric helpers ────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> int:
    return int(_normalize(prediction) == _normalize(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    common      = Counter(pred_tokens) & Counter(gold_tokens)
    num_same    = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Baseline: single Proponent call (no Skeptic loop) ─────────────────────────

_BASELINE_LLM = None

def _get_baseline_llm():
    global _BASELINE_LLM
    if _BASELINE_LLM is None:
        _BASELINE_LLM = ChatGroq(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _BASELINE_LLM


def baseline_answer(question: str) -> str:
    """Single Proponent call with no cross-examination — the comparison baseline."""
    llm = _get_baseline_llm()
    messages = [
        SystemMessage(content="Answer the following factual question concisely and directly."),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# ── Extract short answer from Proponent's structured output ───────────────────

def _extract_conclusion(draft: str) -> str:
    """
    Pulls text after CONCLUSION: label if present.
    Falls back to the last non-empty line of the draft.
    """
    lines = draft.split("\n")
    for i, line in enumerate(lines):
        if "CONCLUSION" in line.upper():
            # Return everything after the colon on this line, or next lines
            after = line.split(":", 1)[-1].strip()
            if after:
                return after
            # Try next line
            if i + 1 < len(lines):
                return lines[i + 1].strip()

    # Fallback: last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return draft.strip()


# ── Main runner ───────────────────────────────────────────────────────────────

def run_hotpotqa_eval(
    n_samples    : int  = None,
    max_turns    : int  = None,
    progress_cb         = None,     # optional callback(i, total, result_row)
) -> dict:
    """
    Run HotpotQA evaluation.

    Parameters
    ----------
    n_samples   : Number of questions to evaluate (default: config.EVAL_SAMPLE_N).
    max_turns   : Cross-examination rounds (default: config.MAX_TURNS).
    progress_cb : Optional callable(i, total, row) for Streamlit progress updates.

    Returns
    -------
    dict with keys: results, summary
    """
    n_samples = n_samples or config.EVAL_SAMPLE_N
    max_turns = max_turns or config.MAX_TURNS

    print(f"\n[HotpotQA] Loading dataset …")
    dataset  = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    samples  = list(dataset.select(range(n_samples)))
    total    = len(samples)

    results  = []
    pipeline_ems, pipeline_f1s = [], []
    baseline_ems, baseline_f1s = [], []

    for i, sample in enumerate(samples):
        question  = sample["question"]
        gold      = sample["answer"]

        print(f"\n[{i+1}/{total}] Q: {question[:80]}…")
        print(f"           Gold: {gold}")

        # ── Pipeline run ──────────────────────────────────────────────────
        t0 = time.time()
        try:
            state     = run_query(question, max_turns=max_turns)
            pipeline_pred = _extract_conclusion(
                state.get("final_answer") or state.get("draft_answer", "")
            )
            pipeline_time = round(time.time() - t0, 2)
        except Exception as e:
            pipeline_pred = ""
            pipeline_time = round(time.time() - t0, 2)
            print(f"  ⚠️  Pipeline error: {e}")

        # ── Baseline run ──────────────────────────────────────────────────
        t0 = time.time()
        try:
            base_pred  = baseline_answer(question)
            base_time  = round(time.time() - t0, 2)
        except Exception as e:
            base_pred = ""
            base_time = round(time.time() - t0, 2)
            print(f"  ⚠️  Baseline error: {e}")

        # ── Compute metrics ───────────────────────────────────────────────
        p_em  = exact_match(pipeline_pred, gold)
        p_f1  = round(token_f1(pipeline_pred, gold), 4)
        b_em  = exact_match(base_pred, gold)
        b_f1  = round(token_f1(base_pred, gold), 4)

        pipeline_ems.append(p_em);  pipeline_f1s.append(p_f1)
        baseline_ems.append(b_em);  baseline_f1s.append(b_f1)

        row = {
            "id"              : i + 1,
            "question"        : question,
            "gold_answer"     : gold,
            "pipeline_pred"   : pipeline_pred,
            "pipeline_em"     : p_em,
            "pipeline_f1"     : p_f1,
            "pipeline_time_s" : pipeline_time,
            "baseline_pred"   : base_pred,
            "baseline_em"     : b_em,
            "baseline_f1"     : b_f1,
            "baseline_time_s" : base_time,
            "turns_used"      : state.get("turn_count", 0) if 'state' in dir() else 0,
        }
        results.append(row)

        print(f"  Pipeline → EM:{p_em}  F1:{p_f1:.3f}  | "
              f"Baseline → EM:{b_em}  F1:{b_f1:.3f}")

        if progress_cb:
            progress_cb(i + 1, total, row)

    # ── Aggregate summary ─────────────────────────────────────────────────
    summary = {
        "dataset"            : "HotpotQA (distractor, validation)",
        "model"              : config.MODEL_NAME,
        "n_samples"          : total,
        "max_turns"          : max_turns,
        "pipeline_em"        : round(sum(pipeline_ems) / total, 4),
        "pipeline_f1"        : round(sum(pipeline_f1s) / total, 4),
        "baseline_em"        : round(sum(baseline_ems) / total, 4),
        "baseline_f1"        : round(sum(baseline_f1s) / total, 4),
        "em_improvement"     : round((sum(pipeline_ems) - sum(baseline_ems)) / total, 4),
        "f1_improvement"     : round((sum(pipeline_f1s) - sum(baseline_f1s)) / total, 4),
        "timestamp"          : datetime.now().isoformat(),
    }

    print(f"\n{'='*60}")
    print(f"  HotpotQA Results ({total} samples)")
    print(f"{'='*60}")
    print(f"  Pipeline  → EM: {summary['pipeline_em']:.3f}  F1: {summary['pipeline_f1']:.3f}")
    print(f"  Baseline  → EM: {summary['baseline_em']:.3f}  F1: {summary['baseline_f1']:.3f}")
    print(f"  EM  Δ = {summary['em_improvement']:+.3f}")
    print(f"  F1  Δ = {summary['f1_improvement']:+.3f}")
    print(f"{'='*60}\n")

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "hotpotqa_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"  Results saved → {out_path}")

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    run_hotpotqa_eval()
