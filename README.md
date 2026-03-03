# Scalable Reflexion: Semantic Memory and Multi-Agent Collaboration for Verbal Reinforcement Learning
This repository contains the official codebase and extensions for the paper **"Scalable Reflexion: Semantic Memory and Multi-Agent Collaboration for Verbal Reinforcement Learning."** 
It builds upon the original modular [Reflexion framework](https://arxiv.org/abs/2303.11366) by introducing three major architectural extensions designed to address long-horizon task limits, collaboration deficits, and policy optimization constraints in autonomous LLM agents.
---
## 🚀 Key Extensions
### 1. Advanced Memory Architectures (Semantic Vector Memory)
Replaces the traditional sliding-window or temporal memory buffer with a persistent, task-isolated **Semantic Vector Database**.
- **Location:** `reflexion/agents/vector.py` & `reflexion/agents/smart.py`
- **Features:** Efficient embedding-based storage, optimized retrieval mechanisms, and zero-shot contextual injection for multi-session tasks. 
- **Experiments:** Validated via the `experiments/extension1_vector_memory` suite (Long-Horizon Benchmarks, Memory Efficiency, and Retrieval Accuracy).
### 2. Multi-Agent Reflexion
Introduces a distributed, multi-agent protocol where heterogeneous LLM agents collaborate, share episodic memories, and co-reflect.
- **Location:** `reflexion/agents/multiagent.py`
- **Features:** Shared semantic memory space, message-passing protocols, and emergent group coordination.
- **Experiments:** Evaluated on collaborative coding tasks against baseline single-agent Reflexion methods.
### 3. Verbal Reinforcement Learning
A hybrid integration of traditional continuous control RL principles with verbal reasoning loops.
- **Location:** `reflexion/agents/rl_reflexion.py`
- **Features:** Formulates linguistic reflections as learned value functions and policy gradients, improving step-by-step optimization in constrained environments.
---
## 🛠️ Repository Structure
```text
├── reflexion/
│   ├── agents/         # Implementations of Vector, Smart, MultiAgent, and RL agents
│   ├── memory/         # Temporal and Vector memory abstractions
│   ├── benchmarks/     # Dataset loaders (e.g., HumanEval)
│   ├── evaluators/     # Code execution and validation logic
│   └── llm.py          # Core LLM API wrappers (OpenRouter/Gemini)
├── experiments/
│   ├── extension1_vector_memory/ # Intensive vector memory ablation scripts
│   ├── run_comparison.py         # Main entrypoint to benchmark all extensions
│   ├── run_humaneval.py          # Standalone HumanEval execution
│   └── visualize_results.py      # Plot generation for the paper
├── analysis/
│   └── ablation_study.py         # Statistical validation metrics logic
├── config.json.template          # Template for required environment configuration
└── README.md
```
---
## ⚙️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/sanihit76201/MTech-Reflexion-Extensions.git
   cd MTech-Reflexion-Extensions
   ```
2. **Install Dependencies:**
   Requires Python 3.9+. Install standard dependencies:
   ```bash
   pip install numpy scipy matplotlib scikit-learn
   # (Include any other specific dependencies like vector DB clients if applicable)
   ```
3. **Configure Environment variables:**
   Copy the `.env.template` (or create a new `.env` file) in the root directory and add your API credentials. **Never commit this file.**
   ```env
   OPENROUTER_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here
   ```
---
## 🔬 Running the Experiments
To run the comprehensive ablation study comparing the Modular Baseline against Extensions 1, 2, and 3, use the primary test script:
```bash
cd experiments
# Test Extension 1 (Isolated Task Semantic Memory)
python run_comparison.py --extension smart --tasks 50
# Test Extension 1 (Full Vector Database)
python run_comparison.py --extension vector --tasks 50
# Test Extension 2 (Multi-Agent Collaboration)
python run_comparison.py --extension multiagent --tasks 50
# Test Extension 3 (Verbal RL)
python run_comparison.py --extension rl --tasks 50
```
*Note: The script will automatically compute and output statistically validated metrics (Pass@1, Pass@3, Cohen's d Effect Size, Paired t-tests) exactly as reported in our paper.*
---
## 📊 Generating Visualizations
After running the comparisons, you can render the performance graphs:
```bash
python experiments/visualize_results.py
```
This will output visual PNGs (like `reflexion_results.png`) demonstrating the quantitative performance deltas between the baselines and the new extensions.
---
### Citation
If you use this code in your research, please cite our paper:
> **"Scalable Reflexion: Semantic Memory and Multi-Agent Collaboration for Verbal Reinforcement Learning"** (2026)
