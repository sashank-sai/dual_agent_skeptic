"""
config.py — Central configuration for the Dual-Agent Skeptic system.
All hyperparameters live here; change once, affects everywhere.
"""

# ── LLM ─────────────────────────────────────────────────────────────────────
MODEL_NAME        = "llama-3.3-70b-versatile"   # Groq free-tier model
TEMPERATURE       = 0.4                  # Moderate creativity; lower = more deterministic
MAX_TOKENS        = 1024                 # Per-call token ceiling

# ── Graph ────────────────────────────────────────────────────────────────────
MAX_TURNS         = 3    # Max Skeptic↔Proponent rounds before Arbiter force-accepts

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_SAMPLE_N     = 50   # Number of HotpotQA validation questions to benchmark
RESULTS_DIR       = "results"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_VERBOSE       = True   # Print turn-by-turn dialogue to stdout
