"""
main.py — CLI entry point for the Dual-Agent Cross-Examination system.

Modes:
  python main.py                          → interactive Q&A session
  python main.py --mode interactive       → same
  python main.py --mode eval              → HotpotQA benchmark (N=config.EVAL_SAMPLE_N)
  python main.py --mode eval --samples 10 → benchmark with 10 samples
  python main.py --mode eval --turns 2    → benchmark with 2 max turns
"""

import argparse
import sys
import os

from dotenv import load_dotenv
load_dotenv()

# ── Validate API key early ────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    print("\n❌  GROQ_API_KEY not found.")
    print("   → Copy .env.example to .env and add your key.")
    print("   → Get a free key at: https://console.groq.com\n")
    sys.exit(1)

import config
from graph.workflow import run_query
from utils.logger   import print_run, log_full_run


def run_interactive():
    """Continuous interactive Q&A loop."""
    print("\n" + "═"*60)
    print("  DUAL-AGENT CROSS-EXAMINATION SYSTEM — Interactive Mode")
    print("  Model  :", config.MODEL_NAME)
    print("  Turns  :", config.MAX_TURNS)
    print("  Type 'quit' or 'exit' to stop.")
    print("═"*60 + "\n")

    while True:
        try:
            query = input("❓  Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!\n")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!\n")
            break

        print("\n⏳  Running cross-examination pipeline …\n")
        try:
            state = run_query(query, max_turns=config.MAX_TURNS)
            print_run(state, verbose=True)
        except Exception as e:
            print(f"\n❌  Error: {e}\n")


def run_eval(n_samples: int, max_turns: int):
    """Batch evaluation on HotpotQA."""
    from evaluation.hotpotqa_runner import run_hotpotqa_eval
    run_hotpotqa_eval(n_samples=n_samples, max_turns=max_turns)


def main():
    parser = argparse.ArgumentParser(
        description="Dual-Agent Skeptic — Cross-Examination LLM System"
    )
    parser.add_argument(
        "--mode", choices=["interactive", "eval"], default="interactive",
        help="Run mode: 'interactive' (default) or 'eval'",
    )
    parser.add_argument(
        "--samples", type=int, default=config.EVAL_SAMPLE_N,
        help=f"Number of HotpotQA samples to evaluate (default: {config.EVAL_SAMPLE_N})",
    )
    parser.add_argument(
        "--turns", type=int, default=config.MAX_TURNS,
        help=f"Max cross-examination turns (default: {config.MAX_TURNS})",
    )
    args = parser.parse_args()

    if args.mode == "eval":
        run_eval(n_samples=args.samples, max_turns=args.turns)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
