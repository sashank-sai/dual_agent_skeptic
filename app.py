"""
app.py — Streamlit Interactive Demo UI
Dual-Agent Cross-Examination System: Adaptive Socratic Interrogation in Multi-Agent LLMs

Run with:  streamlit run app.py
"""

import os
import sys
import json
import time
import threading
from datetime import datetime

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title  = "Dual-Agent Cross-Examination",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ─── Google Font ─── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ─── Dark background ─── */
  .stApp { background: #0a0e1a; color: #e2e8f0; }

  /* ─── Sidebar ─── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #0a0e1a 100%);
    border-right: 1px solid #1e2a45;
  }
  [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* ─── Main header ─── */
  .hero-header {
    background: linear-gradient(135deg, #0f1629 0%, #1a2744 50%, #0f2744 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #9f7aea, #f687b3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
  }
  .hero-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
  }
  .hero-badges {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
  }
  .badge {
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ─── Agent cards ─── */
  .agent-card {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
    border-left: 4px solid;
    position: relative;
    animation: slideIn 0.4s ease-out;
  }
  @keyframes slideIn {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
  }
  .card-proponent {
    background: linear-gradient(135deg, rgba(99,179,237,0.08), rgba(99,179,237,0.03));
    border-color: #63b3ed;
    border-top-right-radius: 12px;
  }
  .card-skeptic {
    background: linear-gradient(135deg, rgba(246,173,85,0.08), rgba(246,173,85,0.03));
    border-color: #f6ad55;
  }
  .card-arbiter {
    background: linear-gradient(135deg, rgba(104,211,145,0.08), rgba(104,211,145,0.03));
    border-color: #68d391;
  }
  .card-final {
    background: linear-gradient(135deg, rgba(159,122,234,0.12), rgba(159,122,234,0.05));
    border-color: #9f7aea;
    border-width: 2px;
  }
  .agent-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
  }
  .label-proponent { color: #63b3ed; }
  .label-skeptic   { color: #f6ad55; }
  .label-arbiter   { color: #68d391; }
  .label-final     { color: #9f7aea; }
  .agent-content {
    color: #e2e8f0;
    font-size: 0.92rem;
    line-height: 1.7;
    white-space: pre-wrap;
  }

  /* ─── JSON interrogation box ─── */
  .json-box {
    background: #0f1629;
    border: 1px solid rgba(246,173,85,0.3);
    border-radius: 8px;
    padding: 1rem;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
  }
  .json-field-label {
    color: #f6ad55;
    font-weight: 500;
  }
  .json-field-value {
    color: #e2e8f0;
  }
  .json-question {
    color: #fbd38d;
    font-weight: 600;
    font-size: 0.9rem;
  }
  .json-criteria li {
    color: #a0aec0;
    margin: 0.2rem 0;
  }

  /* ─── Turn badge ─── */
  .turn-badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 12px;
    padding: 0.15rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin-left: 0.5rem;
    vertical-align: middle;
  }

  /* ─── Metrics cards ─── */
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
  }
  .metric-card {
    background: linear-gradient(135deg, #0f1629, #1a2744);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #63b3ed;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
  }
  .metric-label {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .metric-positive { color: #68d391; }
  .metric-negative { color: #fc8181; }

  /* ─── Query input area ─── */
  .stTextArea textarea {
    background: #0f1629 !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
  }
  .stTextArea textarea:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.1) !important;
  }

  /* ─── Buttons ─── */
  .stButton > button {
    background: linear-gradient(135deg, #3182ce, #9f7aea) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99,179,237,0.3) !important;
  }

  /* ─── Divider ─── */
  hr { border-color: #1e2a45 !important; }

  /* ─── Tab styling ─── */
  .stTabs [data-baseweb="tab-list"] {
    background: #0f1629;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
  }
  .stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #63b3ed !important;
  }

  /* ─── Scrollbar ─── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0a0e1a; }
  ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Import backend (after page config) ───────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()


def _check_api_key():
    return bool(os.getenv("GROQ_API_KEY"))


# ── Session state initialisation ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history       = []   # list of run dicts
if "running" not in st.session_state:
    st.session_state.running       = False
if "eval_results" not in st.session_state:
    st.session_state.eval_results  = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    api_key_input = st.text_input(
        "🔑 Groq API Key",
        value  = os.getenv("GROQ_API_KEY", ""),
        type   = "password",
        help   = "Get a free key at https://console.groq.com",
        key    = "api_key_field",
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

    st.markdown("---")
    st.markdown("### 🎛️ Model Settings")

    model_choice = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama-3.1-8b-instant"],
        index = 0,
        key   = "model_select",
    )

    max_turns = st.slider(
        "Max Cross-Examination Turns",
        min_value = 1, max_value = 5, value = 3, step = 1,
        help      = "Number of Skeptic→Proponent rounds before Arbiter force-accepts.",
        key       = "max_turns_slider",
    )

    temperature = st.slider(
        "Temperature",
        min_value = 0.0, max_value = 1.0, value = 0.4, step = 0.05,
        key       = "temperature_slider",
    )

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **Dual-Agent Self-Correction**
    using Structured Socratic Cross-Examination.

    - 🎭 **Proponent** — drafts & defends answers
    - 🔍 **Skeptic** — JSON interrogation of weakest premise
    - ⚖️ **Arbiter** — routes loop or accepts final answer

    Built with **LangGraph** + **Groq** (`llama-3.3-70b-versatile`)
    """)

    if st.button("🗑️ Clear History", key="clear_btn"):
        st.session_state.history = []
        st.rerun()


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🔍 Dual-Agent Cross-Examination</div>
  <div class="hero-subtitle">
    Adaptive Socratic Interrogation in Multi-Agent LLMs — Research Demo
  </div>
  <div class="hero-badges">
    <span class="badge">LangGraph</span>
    <span class="badge">Groq API</span>
    <span class="badge">llama-3.3-70b-versatile</span>
    <span class="badge">HotpotQA</span>
    <span class="badge">Socratic Method</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_interactive, tab_eval, tab_history = st.tabs([
    "🎭 Interactive Demo",
    "📊 HotpotQA Evaluation",
    "📜 Run History",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — INTERACTIVE DEMO
# ═══════════════════════════════════════════════════════════════
with tab_interactive:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 💬 Ask a Research Question")
        st.markdown(
            "<p style='color:#64748b;font-size:0.85rem;margin-top:-0.5rem;'>"
            "The Skeptic will cross-examine the Proponent's answer turn by turn.</p>",
            unsafe_allow_html=True,
        )

        example_qs = [
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
            "The Oberoi family is part of a hotel company that has a head office in what city?",
            "Did Aristotle use a laptop?",
            "Which magazine was started first — Arthur's Magazine or First for Women?",
        ]

        selected_example = st.selectbox(
            "💡 Load an example question",
            ["(type your own below)"] + example_qs,
            key="example_select",
        )

        default_q = "" if selected_example.startswith("(type") else selected_example
        query = st.text_area(
            "Your question",
            value       = default_q,
            height      = 100,
            placeholder = "e.g. Were Scott Derrickson and Ed Wood of the same nationality?",
            key         = "query_input",
            label_visibility = "collapsed",
        )

        run_col, _ = st.columns([1, 2])
        with run_col:
            run_btn = st.button(
                "⚡ Run Cross-Examination",
                disabled = (not query.strip() or st.session_state.running),
                key      = "run_btn",
                use_container_width = True,
            )

    with col_right:
        st.markdown("### 🔄 Live Dialogue")

        if not _check_api_key():
            st.warning("⚠️ Please enter your Groq API key in the sidebar to get started.")

        dialogue_placeholder = st.empty()

        if run_btn and query.strip() and _check_api_key():
            st.session_state.running = True

            # ── Update config from sidebar ────────────────────────────────
            import config as cfg
            cfg.MODEL_NAME   = model_choice
            cfg.TEMPERATURE  = temperature
            cfg.MAX_TURNS    = max_turns

            from graph.workflow import run_query

            with st.spinner("⏳ Running Socratic cross-examination …"):
                t_start = time.time()
                try:
                    final_state = run_query(query.strip(), max_turns=max_turns)
                    elapsed     = round(time.time() - t_start, 1)

                    # ── Store run in history ──────────────────────────────
                    run_record = {
                        "query"               : query.strip(),
                        "final_answer"        : final_state.get("final_answer") or final_state.get("draft_answer", ""),
                        "conversation_history": final_state.get("conversation_history", []),
                        "turn_count"          : final_state.get("turn_count", 0),
                        "max_turns"           : max_turns,
                        "elapsed_s"           : elapsed,
                        "timestamp"           : datetime.now().strftime("%H:%M:%S"),
                    }
                    st.session_state.history.insert(0, run_record)

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state.running = False
                    st.stop()

            st.session_state.running = False

        # ── Render last run ───────────────────────────────────────────────
        if st.session_state.history:
            last = st.session_state.history[0]
            history = last["conversation_history"]

            turn_counter = 1
            for entry in history:
                role    = entry["role"]
                content = entry["content"]

                if role == "proponent":
                    st.markdown(
                        f'<div class="agent-card card-proponent">'
                        f'<div class="agent-label label-proponent">'
                        f'⚡ Proponent <span class="turn-badge">Turn {turn_counter}</span>'
                        f'</div>'
                        f'<div class="agent-content">{content}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                elif role == "skeptic":
                    # Pretty-render JSON
                    try:
                        d = json.loads(content)
                        criteria_html = "".join(
                            f"<li>{c}</li>" for c in d.get("expected_defense_criteria", [])
                        )
                        skeptic_html = (
                            f'<div class="json-box">'
                            f'<div><span class="json-field-label">Weakest Premise: </span>'
                            f'<span class="json-field-value">{d.get("weakest_premise","")}</span></div>'
                            f'<div style="margin-top:0.4rem"><span class="json-field-label">Type: </span>'
                            f'<span class="json-field-value">{d.get("interrogation_type","")}</span></div>'
                            f'<div style="margin-top:0.6rem;padding:0.5rem;background:rgba(246,173,85,0.06);border-radius:6px">'
                            f'<span class="json-question">❓ {d.get("question","")}</span></div>'
                            f'<div style="margin-top:0.5rem"><span class="json-field-label">Defense criteria:</span>'
                            f'<ul class="json-criteria">{criteria_html}</ul></div>'
                            f'</div>'
                        )
                    except Exception:
                        skeptic_html = f'<div class="agent-content">{content}</div>'

                    st.markdown(
                        f'<div class="agent-card card-skeptic">'
                        f'<div class="agent-label label-skeptic">🔍 Skeptic — Interrogation</div>'
                        f'{skeptic_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                elif role == "arbiter":
                    st.markdown(
                        f'<div class="agent-card card-arbiter">'
                        f'<div class="agent-label label-arbiter">⚖️ Arbiter</div>'
                        f'<div class="agent-content">{content}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    turn_counter += 1

            # ── Final answer ──────────────────────────────────────────────
            st.markdown(
                f'<div class="agent-card card-final">'
                f'<div class="agent-label label-final">✅ Final Accepted Answer</div>'
                f'<div class="agent-content">{last["final_answer"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Run stats ─────────────────────────────────────────────────
            st.markdown(
                f'<div style="display:flex;gap:1.5rem;margin-top:1rem;color:#64748b;font-size:0.8rem;">'
                f'<span>⏱ {last["elapsed_s"]}s</span>'
                f'<span>🔄 {last["turn_count"]} turn(s)</span>'
                f'<span>🕐 {last["timestamp"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="color:#475569;text-align:center;padding:3rem;border:1px dashed #1e3a5f;border-radius:12px;">'
                '🔍 Enter a question and click <b>Run Cross-Examination</b> to begin.'
                '</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════
# TAB 2 — HOTPOTQA EVALUATION
# ═══════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 📊 HotpotQA Benchmark Evaluation")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;margin-top:-0.5rem;'>"
        "Compares the full pipeline (Proponent + Skeptic + Arbiter) against a "
        "single-agent baseline on multi-hop factual questions.</p>",
        unsafe_allow_html=True,
    )

    eval_col1, eval_col2 = st.columns([1, 1], gap="large")

    with eval_col1:
        n_samples_eval = st.number_input(
            "Number of evaluation samples",
            min_value = 5, max_value = 200, value = 20, step = 5,
            key       = "n_samples_eval",
        )
        max_turns_eval = st.slider(
            "Max turns (eval)",
            min_value = 1, max_value = 5, value = 3, step = 1,
            key       = "max_turns_eval",
        )

        eval_btn = st.button(
            "🚀 Run HotpotQA Evaluation",
            disabled = not _check_api_key(),
            key      = "eval_btn",
            use_container_width = True,
        )

        st.info(
            "⚠️ Each sample makes multiple LLM calls. "
            f"{n_samples_eval} samples ≈ {n_samples_eval * max_turns_eval * 3} API calls. "
            "Free Groq tier has rate limits — start small (5–10).",
            icon="ℹ️"
        )

    with eval_col2:
        if st.session_state.eval_results:
            s = st.session_state.eval_results["summary"]
            st.markdown("#### 📈 Latest Results")

            em_delta  = s["em_improvement"]
            f1_delta  = s["f1_improvement"]
            em_color  = "metric-positive" if em_delta >= 0 else "metric-negative"
            f1_color  = "metric-positive" if f1_delta >= 0 else "metric-negative"

            st.markdown(f"""
            <div class="metric-grid">
              <div class="metric-card">
                <div class="metric-value">{s['pipeline_em']:.2f}</div>
                <div class="metric-label">Pipeline EM</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">{s['pipeline_f1']:.2f}</div>
                <div class="metric-label">Pipeline F1</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">{s['baseline_em']:.2f}</div>
                <div class="metric-label">Baseline EM</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">{s['baseline_f1']:.2f}</div>
                <div class="metric-label">Baseline F1</div>
              </div>
              <div class="metric-card">
                <div class="metric-value {em_color}">{em_delta:+.3f}</div>
                <div class="metric-label">EM Δ</div>
              </div>
              <div class="metric-card">
                <div class="metric-value {f1_color}">{f1_delta:+.3f}</div>
                <div class="metric-label">F1 Δ</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    if eval_btn and _check_api_key():
        import config as cfg
        cfg.MODEL_NAME   = model_choice
        cfg.TEMPERATURE  = temperature

        from evaluation.hotpotqa_runner import run_hotpotqa_eval

        progress_bar  = st.progress(0)
        status_text   = st.empty()
        results_store = {"data": None}

        def progress_cb(i, total, row):
            pct = int(i / total * 100)
            progress_bar.progress(pct)
            status_text.markdown(
                f"<span style='color:#94a3b8;font-size:0.85rem'>"
                f"[{i}/{total}] Pipeline F1: {row['pipeline_f1']:.3f} | "
                f"Baseline F1: {row['baseline_f1']:.3f}"
                f"</span>",
                unsafe_allow_html=True,
            )

        with st.spinner("Running evaluation …"):
            eval_data = run_hotpotqa_eval(
                n_samples   = int(n_samples_eval),
                max_turns   = max_turns_eval,
                progress_cb = progress_cb,
            )

        progress_bar.progress(100)
        status_text.success("✅ Evaluation complete!")
        st.session_state.eval_results = eval_data
        st.rerun()

    # ── Show per-question results table ──────────────────────────────────
    if st.session_state.eval_results:
        import pandas as pd
        results_list = st.session_state.eval_results["results"]
        df = pd.DataFrame(results_list)[[
            "id", "question", "gold_answer",
            "pipeline_pred", "pipeline_em", "pipeline_f1",
            "baseline_pred", "baseline_em", "baseline_f1",
            "turns_used",
        ]]
        df.columns = [
            "#", "Question", "Gold",
            "Pipeline Answer", "P-EM", "P-F1",
            "Baseline Answer", "B-EM", "B-F1",
            "Turns",
        ]
        st.markdown("#### 🗂️ Per-Question Results")
        st.dataframe(df, use_container_width=True, height=400)

        # ── Download ──────────────────────────────────────────────────────
        json_str = json.dumps(st.session_state.eval_results, indent=2)
        st.download_button(
            "⬇️ Download JSON Results",
            data      = json_str,
            file_name = f"hotpotqa_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime      = "application/json",
            key       = "download_eval",
        )


# ═══════════════════════════════════════════════════════════════
# TAB 3 — RUN HISTORY
# ═══════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📜 Session Run History")

    if not st.session_state.history:
        st.markdown(
            '<div style="color:#475569;text-align:center;padding:3rem;border:1px dashed #1e3a5f;border-radius:12px;">'
            'No runs yet. Go to the Interactive Demo tab to get started.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for idx, run in enumerate(st.session_state.history):
            with st.expander(
                f"[{run['timestamp']}] {run['query'][:80]}{'…' if len(run['query'])>80 else ''}",
                expanded = (idx == 0),
            ):
                st.markdown(f"**Query:** {run['query']}")
                st.markdown(f"**Turns used:** {run['turn_count']} / {run['max_turns']}  |  "
                            f"**Time:** {run['elapsed_s']}s")

                st.markdown("**Final Answer:**")
                st.markdown(
                    f'<div class="agent-card card-final">'
                    f'<div class="agent-content">{run["final_answer"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if st.checkbox("Show full dialogue", key=f"hist_dial_{idx}"):
                    for entry in run["conversation_history"]:
                        role = entry["role"]
                        css  = {"proponent":"card-proponent","skeptic":"card-skeptic","arbiter":"card-arbiter"}.get(role,"")
                        lbl  = {"proponent":"⚡ PROPONENT","skeptic":"🔍 SKEPTIC","arbiter":"⚖️ ARBITER"}.get(role, role.upper())
                        st.markdown(
                            f'<div class="agent-card {css}">'
                            f'<div class="agent-label">{lbl}</div>'
                            f'<div class="agent-content">{entry["content"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        # Export history
        if st.button("⬇️ Export Session History", key="export_history"):
            st.download_button(
                "Download History JSON",
                data      = json.dumps(st.session_state.history, indent=2),
                file_name = f"session_history_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime      = "application/json",
                key       = "dl_history",
            )
