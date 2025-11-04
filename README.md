<div align="center">

# 🔬 HypoSpace

### *Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination*

[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2510.15614)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="figs/logo.png" alt="Overview">
</p>

## 🙏 Thanks 🙏
This project originates from NUS CEG EEC4300 and we gratefully acknowledge the foundational groundwork provided by [CTT-Pavilion/_HypoSpace](https://github.com/CTT-Pavilion/_HypoSpace).


## 📖 About

> **TL;DR**: HypoSpace evaluates how well LLMs generate *diverse sets* of valid hypotheses in underdetermined scientific problems, not just single correct answers.

### 🌈 The Challenge

As language models are increasingly used in scientific workflows, evaluating their ability to propose **sets of explanations**—not just a single correct answer—becomes critical. Many scientific problems are **underdetermined**: multiple, mechanistically distinct hypotheses are consistent with the same observations.

### 😊Original Solution

We introduce **HypoSpace**, a diagnostic suite that treats LLMs as samplers of finite hypothesis sets and measures three complementary indicators:

| Metric | Symbol | What It Measures |
|--------|--------|------------------|
| **🎯 Validity** | *V* | Precision of proposals consistent with observations |
| **✨ Uniqueness** | *U* | Non-redundancy among proposals |
| **📈 Recovery** | *R* | Coverage of the enumerated admissible set |

### 🤔Key Findings

Across instruction-tuned and reasoning-focused models, **Validity** often remains high while **Uniqueness** and **Recovery** degrade as the admissible space grows, revealing **mode collapse** that is invisible to correctness-only metrics.

> 💡 HypoSpace offers a controlled probe—rather than a leaderboard—for methods that explicitly explore and cover admissible explanation spaces.

### ☝️ Our group solutions

1. **🧬 Causal Graphs** Innovative LLM causal discovery: constraint-guided, hypothesis bank, structured prompts.

2. **📦 3D Voxel Reconstruction** Innovative LLM 3D inference: loss-guided, annealing search, ensemble.

3. **🔀 Boolean Genetic Interactions** Innovative solver-guided LLM discovery: library queue, synthesis hints, enforced reproduction.

> 🧯 To increase task **complexity**, we chose a larger number of **nodes and blocks**, among other **more difficult options**, when generating the dataset.

### ✨Results

</div>

<div style="text-align:center;">
  <figure style="display:block; margin:12px auto; text-align:center;">
    <img src="figs/casual.png" alt="Causal" height="200" style="display:block; margin:0 auto;" />
    <figcaption style="margin-top:6px; font-size:0.95em;">🧬 Causal Graphs</figcaption>
  </figure>

  <figure style="display:block; margin:12px auto; text-align:center;">
    <img src="figs/3d.png" alt="3D Reconstruction" height="200" style="display:block; margin:0 auto;" />
    <figcaption style="margin-top:6px; font-size:0.95em;">📦 3D Reconstruction</figcaption>
  </figure>

  <figure style="display:block; margin:12px auto; text-align:center;">
    <img src="figs/bool.png" alt="Boolean Logic" height="200" style="display:block; margin:0 auto;" />
    <figcaption style="margin-top:6px; font-size:0.95em;">🔀 Boolean Logic</figcaption>
  </figure>
</div>


### Three Structured Domains

We instantiate HypoSpace in three domains with deterministic validators and exactly enumerated hypothesis spaces:

1. **🧬 Causal Graphs** — from perturbations
2. **📦 3D Voxel Reconstruction** — gravity-constrained from top-down projections  
3. **🔀 Boolean Genetic Interactions** — logical function discovery


---

## 📁 Repository Structure

```
📂 HypoSpace/
│
├── 📦 3d/                                  ← 3D Voxel Reconstruction Domain
│   ├── 🔧 generate_3d_dataset_complete.py  • Dataset generator
│   ├── 🚀 run_3d_benchmark.py              • Benchmark runner
│   ├── 📚 modules/                         • LLM interface & models
│   │   ├── llm_interface.py
│   │   └── models.py
│   └── ⚙️  config/
│       └── config_gpt4o.yaml               • Configuration file
│
├── 🔀 boolean/                             ← Boolean Genetic Interactions
│   ├── 🔧 boolean_dataset.py               • Dataset generator
│   ├── 🚀 boolean_benchmark.py             • Benchmark runner
│   ├── 📚 modules/
│   │   ├── llm_interface.py
│   │   └── models.py
│   └── ⚙️  config/
│       └── config_gpt4o.yaml
│
└── 🧬 causal/                              ← Causal Graph Discovery
    ├── 🔧 generate_causal_dataset.py       • Dataset generator (small)
    ├── 🔧 generate_causal_dataset_for_large.py  • Dataset generator (large)
    ├── 🚀 run_causal_benchmark.py          • Benchmark runner
    ├── 📚 modules/
    │   ├── llm_interface.py
    │   └── models.py
    └── ⚙️  config/
        └── config_gpt4o.yaml
```

---

## 🚀 Quick Start

### Step 1️⃣: Configure Your LLM

Edit the YAML config files in each domain's `config/` folder:

**What you can customize:**
- 🤖 LLM provider and model
- 🌡️ Temperature settings
- 📂 Output paths
- 💾 Checkpoint directories

**Example:** `config/config_gpt4o.yaml`

```yaml
llm:
  type: openrouter              # Options: openai, anthropic, openrouter
  models:
    openrouter: "deepseek/deepseek-chat-v3-0324"
  api_keys:
    openrouter: "your-api-key"  # ⚠️ Replace with your actual API key
  temperature: 0.7              # 0.0 = deterministic, 1.0 = creative

benchmark:
  checkpoint: "checkpoints"     # Resume interrupted runs
  verbose: true                 # Print detailed logs
  output_pattern: "results/{dataset_name}_{model}.json"
```

### Step 2️⃣: Generate Datasets

Each domain has its own dataset generator. Here are examples for all three:

<details>
<summary><b>🧬 Causal Graphs</b> (click to expand)</summary>

```bash
cd causal
python generate_causal_dataset.py \
  --nodes 4 \
  --seed 33550336 \
  --output "datasets/node03/n3_all_observations.json"
```

**Parameters:**
- `--nodes`: Number of nodes in graphs (3, 4, 5, etc.)
- `--seed`: Random seed for reproducibility
- `--output`: Path to save dataset JSON

</details>

<details>
<summary><b>📦 3D Voxel Reconstruction</b> (click to expand)</summary>

```bash
cd 3d
python generate_3d_dataset_complete.py \
  --grid-size 3 \
  --max-height 3 \
  --max-blocks 3 \
  --fixed \
  --seed 33550336 \
  --output "datasets/3d_grid3_h3.json"
```

**Parameters:**
- `--grid-size`: Grid dimensions (e.g., 3 for 3×3)
- `--max-height`: Maximum structure height
- `--max-blocks`: Maximum number of blocks in top view
- `--fixed`: If set, generate only structures with exactly max-blocks blocks, else from 1 to max-blocks
- `--output`: Output file path

</details>

<details>
<summary><b>🔀 Boolean Logic</b> (click to expand)</summary>

```bash
cd boolean
python boolean_dataset.py \
  --operators basic \
  --max-depth 2 \
  --output 'datasets/boolean_2var.json' \
  --seed 33550336 \
```

**Parameters:**
- `--operators`: Allowed Boolean operators: choices=['basic', 'extended', 'full']
- `--max-depth`: Maximum expression depth
- `--output`: Output JSON file

</details>

### Step 3️⃣: Run Benchmarks

Run the benchmark for your chosen domain:

<details>
<summary><b>🧬 Causal Benchmark</b></summary>

```bash
cd causal
python run_causal_benchmark.py \
  --dataset "datasets/node03/n3_all_observations.json" \
  --config "config/config_gpt4o.yaml" \
  --n-samples 30 \
  --query-multiplier 1.0 \
  --seed 33550336
```

**Run in background with logging:**
```bash
nohup python -u run_causal_benchmark.py \
  --dataset "datasets/node03/n3_all_observations.json" \
  --config "config/config_gpt4o.yaml" \
  --n-samples 30 \
  --query-multiplier 1.0 \
  --seed 33550336 > logs/causal_gpt4o.log 2>&1 &
```

</details>

<details>
<summary><b>📦 3D Benchmark</b></summary>

```bash
cd 3d
python run_3d_benchmark.py \
  --dataset "datasets/3d_grid3_h3.json" \
  --config "config/config_gpt4o.yaml" \
  --n-samples 30 \
  --query-multiplier 1.0 \
  --seed 33550336
```

</details>

<details>
<summary><b>🔀 Boolean Benchmark</b></summary>

```bash
cd boolean
python boolean_benchmark.py \
  --dataset "datasets/boolean_2var.json" \
  --config "config/config_gpt4o.yaml" \
  --n-samples 30 \
  --query-multiplier 1.0 \
  --seed 33550336
```

</details>

**Common Parameters:**
- `--dataset`: Path to generated dataset
- `--config`: Configuration YAML file
- `--n-samples`: Number of observation sets to evaluate
- `--query-multiplier`: Multiplier for queries per task
- `--seed`: Random seed for reproducibility

### Step 4️⃣: Analyze Results

Results are automatically saved as JSON files in the `results/` directory.

**What's included:**

```json
{
  "metadata": {
    "model": "openai/gpt-4o",
    "dataset": "causal_n3",
    "n_samples": 30,
    "timestamp": "2025-10-17T12:00:00"
  },
  "aggregate_metrics": {
    "mean_validity": 0.92,      // 🎯 How many proposals are valid
    "mean_uniqueness": 0.78,    // ✨ How diverse are the proposals
    "mean_recovery": 0.65,      // 📈 Coverage of solution space
    "std_validity": 0.08,
    "std_uniqueness": 0.12,
    "std_recovery": 0.15
  },
  "results": [/* detailed per-sample results */]
}
```

**Understanding the Metrics:**

| Metric | Range | Good Score | Interpretation |
|--------|-------|------------|----------------|
| 🎯 **Validity** | 0-1 | > 0.90 | Model proposes correct hypotheses |
| ✨ **Uniqueness** | 0-1 | > 0.80 | Model avoids redundant proposals |
| 📈 **Recovery** | 0-1 | > 0.80 | Model explores solution space well |

---

## 📊 Supported Models

| Provider | Example Models | Config Type |
|----------|----------------|-------------|
| **OpenAI** | GPT-4o, GPT-4-turbo, GPT-3.5 | `openai` |
| **OpenRouter** | Any model via OpenRouter | `openrouter` |

---

## 📝 Citation

If you use HypoSpace in your research, please cite:

```bibtex
@article{chen2025hypospace,
  title={HypoSpace: Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination},
  author={Chen, Tingting and Lin, Beibei and Yuan, Zifeng and Zou, Qiran and He, Hongyu and Ong, Yew-Soon and Goyal, Anirudh and Liu, Dianbo},
  journal={arXiv preprint arXiv:2510.15614},
  year={2025}
}
```

---

## 📄 License

This project is released under the MIT License.

---

<div align="center">

**Built with ❤️ for scientific discovery**

⭐ Star us on GitHub • 🐛 Report issues • 💡 Suggest features

</div>
