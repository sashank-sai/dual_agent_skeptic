# 🔍 Dual-Agent Skeptic: Cross-Examination LLM System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.0+-green.svg)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)

**Dual-Agent Skeptic** is an advanced, multi-agent AI system designed to improve large language model (LLM) reasoning through **adaptive Socratic interrogation**. By structuring a debate between specialized agents, the system systematically pressure-tests hypotheses, identifies weak premises, and self-corrects before arriving at a final conclusion.

## ✨ Key Features

- **Multi-Agent Architecture**: Powered by LangGraph, coordinating three distinct roles (Proponent, Skeptic, Arbiter).
- **Socratic Interrogation**: The Skeptic agent specifically targets the weakest premise of a drafted answer using structured JSON criteria.
- **HotpotQA Evaluation**: Built-in benchmarking suite to evaluate factual accuracy on complex, multi-hop reasoning questions.
- **Beautiful Streamlit UI**: A rich, interactive web interface for real-time visualization of the cross-examination process.
- **CLI Mode**: A terminal-based interactive Q&A session and headless evaluation mode.
- **Groq Integration**: Optimized for fast inference using Groq's high-speed API.

## 🏗️ How It Works

1. **🎭 Proponent**: Drafts an initial answer to the user's question and defends it against critiques.
2. **🔍 Skeptic**: Analyzes the Proponent's answer, isolates the weakest premise, and issues a targeted, structured interrogation to challenge it.
3. **⚖️ Arbiter**: Evaluates the debate. It either routes the conversation back for another round of cross-examination or force-accepts the final answer if a consensus is reached (or the maximum turn limit is hit).

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API Key](https://console.groq.com)

### Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/YOUR_USERNAME/dual_agent_skeptic.git
   cd dual_agent_skeptic
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**:
   Copy the example environment file and add your Groq API key:
   ```bash
   cp .env.example .env
   ```
   *Open `.env` and replace `your_groq_api_key_here` with your actual API key.*

## 💻 Usage

### 1. Interactive Web UI (Streamlit)
Experience the system visually with real-time agent dialogue, metrics, and evaluation tools.
```bash
streamlit run app.py
```

### 2. Command Line Interface (CLI)
**Interactive Mode:**
```bash
python main.py --mode interactive
```

**Evaluation Mode (HotpotQA):**
Run automated benchmarks to compare the dual-agent pipeline against a single-agent baseline.
```bash
python main.py --mode eval --samples 50 --turns 3
```

## 📂 Project Structure

```text
dual_agent_skeptic/
├── agents/             # Agent definitions (Proponent, Skeptic, Arbiter)
├── evaluation/         # Benchmarking scripts (HotpotQA evaluator)
├── graph/              # LangGraph workflow definitions
├── utils/              # Helper functions and logger
├── app.py              # Streamlit Web UI application
├── main.py             # CLI entry point
├── config.py           # Global hyperparameters and settings
├── requirements.txt    # Python dependencies
└── .env                # Environment variables (API Keys)
```


