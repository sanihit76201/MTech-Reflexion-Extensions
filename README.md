# Scalable Reflexion: Semantic Memory and Multi-Agent Collaboration for Verbal Reinforcement Learning

> **Dagani Jesu Sanihit** · Manipal Institute of Technology, Udupi, Karnataka, India  
> *Transactions on Machine Learning Research (2026)*

This repository contains the official implementation for the paper **"Scalable Reflexion: Semantic Memory and Multi-Agent Collaboration for Verbal Reinforcement Learning"**, which proposes two orthogonal extensions to the [Reflexion](https://arxiv.org/abs/2303.11366) framework.

---

## Overview

[Reflexion](https://arxiv.org/abs/2303.11366) improves LLM performance through verbal self-reflection and iterative trial refinement. This work addresses two structural limitations of the original framework:

| Limitation | Our Solution |
|---|---|
| FIFO sliding window discards semantically relevant older memories | **Extension 1:** `VectorEpisodicMemory` — Sentence-BERT embeddings with cosine-similarity retrieval |
| Single agent conflates generation, critique, and verification | **Extension 2:** Generator–Critic–Verifier multi-agent pipeline with shared memory pool |

### Key Results (164 HumanEval tasks, Gemini 2.5 Flash)

| Agent | Pass@3 | Pass@1 | Avg Trials |
|---|---|---|---|
| Modular Baseline | 89.0% | 81.7% | 1.10 |
| Vector Reflexion (E1) | 92.7% | 87.2% | 1.09 |
| Multi-Agent Reflexion (E2) | **96.3%** | **93.9%** | **1.03** |

On the long-horizon memory benchmark (9 distractor tasks):

| Memory Type | Session-5 Success | Dependency Recall |
|---|---|---|
| Temporal (FIFO) | 0% | 50% |
| VectorEpisodicMemory | **100%** | **100%** |

---

## Repository Structure

```
MTech-Reflexion-Extensions/
├── reflexion/
│   ├── agents/
│   │   ├── base.py              # ReflexionAgent (modular baseline)
│   │   ├── original.py          # OriginalReflexionAgent
│   │   ├── vector.py            # VectorReflexionAgent (Extension 1)
│   │   ├── multiagent.py        # MultiAgentReflexion (Extension 2)
│   │   └── __init__.py
│   ├── memory/
│   │   ├── base.py              # BaseMemory abstract class
│   │   ├── temporal.py          # TemporalMemory (FIFO baseline)
│   │   ├── vector.py            # VectorEpisodicMemory (Extension 1)
│   │   └── __init__.py
│   ├── benchmarks/
│   │   ├── humaneval.py         # HumanEval dataset loader
│   │   └── __init__.py
│   ├── evaluators/
│   │   ├── code.py              # ObjectiveCodeEvaluator
│   │   └── __init__.py
│   ├── reflection/
│   │   ├── config.py
│   │   ├── llm.py
│   │   └── memory.py
│   └── llm.py                   # BaseLLMModel with exponential backoff
├── experiments/
│   ├── extension1_vector_memory/  # Long-horizon & memory efficiency benchmarks
│   ├── run_comparison.py          # Main benchmark entrypoint (Extensions 1 & 2)
│   ├── run_humaneval.py           # Standalone HumanEval runner
│   ├── make_results_table.py      # Generate paper tables
│   └── visualize_results.py      # Generate paper figures
├── results/                       # JSON output from all experiments
├── HumanEval.jsonl.gz             # HumanEval dataset (164 tasks)
├── config.json.template           # API configuration template
├── reflexion_framework.ipynb      # Interactive notebook walkthrough
└── README.md
```

---

## Installation

Requires **Python 3.9+**.

```bash
git clone https://github.com/sanihit76201/MTech-Reflexion-Extensions.git
cd MTech-Reflexion-Extensions
pip install numpy scipy scikit-learn sentence-transformers
```

### API Configuration

Copy the template and fill in your credentials:

```bash
cp config.json.template .env
```

Edit `.env`:

```
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=google/gemini-2.5-flash
GEMINI_API_BASE=https://openrouter.ai/api/v1
RATE_LIMIT_DELAY=0.5
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

---

## Running the Experiments

### Extension 1 — Semantic Vector Memory

```bash
cd experiments
python run_comparison.py --extension vector --tasks 164
```

### Extension 2 — Multi-Agent Collaboration

```bash
cd experiments
python run_comparison.py --extension multiagent --tasks 164
```

Both commands run the **Modular Baseline**, **Original Reflexion**, and the selected extension in sequence, then print:
- Pass@3, Pass@1, average trials
- Paired t-test (p-value), Cohen's d effect size
- 95% Wilson confidence intervals

Results are saved to `results/extension{N}_{type}_agent.json`.

### Long-Horizon Memory Benchmark (Extension 1)

```bash
cd experiments/extension1_vector_memory
python run_long_horizon_benchmark.py
```

This runs the 5-session, 13-task benchmark with 9 injected distractor tasks and reports dependency recall and Session-5 success rate across 3 independent trials.

### Memory Efficiency Scaling

```bash
cd experiments/extension1_vector_memory
python run_memory_efficiency.py
```

Reports retrieval latency across pool sizes from 100 to 50,000 entries.

### Visualizations

```bash
cd experiments
python visualize_results.py
python make_results_table.py
```

---

## Architecture

### Extension 1: VectorEpisodicMemory

Replaces the FIFO sliding window with Sentence-BERT (`all-MiniLM-L6-v2`, 384-dim) embeddings. At retrieval time, cosine similarity is computed between the current task query and all stored reflections, returning the top-k most semantically relevant ones — regardless of when they were stored.

```
sim(q, rᵢ) = φ(q)ᵀ φ(rᵢ) / (‖φ(q)‖ · ‖φ(rᵢ)‖)
```

Retrieval overhead is ~14 ms constant up to 50,000 entries (< 2.8% of LLM call latency).

### Extension 2: Multi-Agent Reflexion

Three specialised agents share a single `VectorEpisodicMemory` pool:

```
Task → [Generator] → candidate code c
              ↓
         [Critic] → structured critique (no code generation)
              ↓
        [Verifier] → final submission c*
              ↓
         Evaluator → Pass / Fail → reflect (all three agents write back)
```

Each agent retrieves memories using a role-conditioned query (e.g. `[Critic] + task prompt`) to bias retrieval toward role-relevant past experience. On failure, all three agents write role-prefixed reflections (`[Generator]`, `[Critic]`, `[Verifier]`) back to the shared pool.

---

## Hyperparameters

| Component | Parameter | Value |
|---|---|---|
| LLM | Model | `google/gemini-2.5-flash` |
| LLM | Temperature | 0.7 |
| LLM | Max tokens | 2048 |
| LLM | Inter-request delay | 0.5 s |
| Backoff | Max retries / Initial delay / Factor | 5 / 5 s / 2.5× |
| TemporalMemory | Buffer size / Top-k | 10 / 3 |
| VectorEpisodicMemory | Encoder | `all-MiniLM-L6-v2` |
| VectorEpisodicMemory | Embedding dim / Max pool / Top-k | 384 / 1000 / 5 |
| MultiAgentReflexion | Top-k per agent | 3 |
| HumanEval | Max trials / Timeout | 3 / 10 s |
| Long-horizon benchmark | Sessions / Tasks / Distractors / Trials | 5 / 13 / 9 / 3 |

---

## Statistical Validation

All HumanEval comparisons use:
- **Paired t-test** on per-task binary success indicators (164 paired observations)
- **Cohen's d** effect size on pooled standard deviation
- **95% Wilson score confidence intervals** (`statsmodels`)

| Comparison | ΔPass@3 | p-value | Cohen's d |
|---|---|---|---|
| E1 vs Baseline | +3.7 pp | 0.033* | 0.127 |
| E2 vs Baseline | +7.9 pp | <0.001*** | 0.301 |

---

## Reproducibility

No hyperparameter search was performed. All values were set a priori based on Reflexion paper defaults where applicable. The long-horizon benchmark is fully procedural and deterministic given the fixed task sequence. Statistical tests use `statsmodels.stats.proportion.proportion_confint` with `method='wilson'`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{sanihit2026scalablereflexion,
  title     = {Scalable Reflexion: Semantic Memory and Multi-Agent Collaboration for Verbal Reinforcement Learning},
  author    = {Dagani Jesu Sanihit},
  journal   = {Transactions on Machine Learning Research},
  year      = {2026}
}
```

---

## Acknowledgements

This work builds upon [Reflexion](https://arxiv.org/abs/2303.11366) by Shinn et al. (NeurIPS 2023). Sentence-BERT embeddings use the [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model from Reimers & Gurevych (EMNLP 2019). All LLM calls use Google Gemini 2.5 Flash via OpenRouter.

