# **Benchmarking RL via Converse Optimality**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## **Overview**

This repository contains the official implementation accompanying
**“Benchmarking RL via Converse Optimality: Generating Systems with Known Optimal Policies”**
(submitted for peer review).

The project provides a **scientifically rigorous benchmarking framework** for reinforcement learning, built around *converse-optimality constructions* that generate dynamical systems with:

* **provably known optimal policies** (`π*`)
* **closed-form optimal value functions** (`V*`)
* **stochastic nonlinear control-affine dynamics**
* **tunable difficulty & conditioning**

This enables ground-truth evaluation of RL algorithms in ways that classical environments cannot:
true optimality gaps, exact regret, and precise sensitivity analysis.

---

## **Motivation**

Standard RL benchmarks do not provide access to the optimal value function or optimal feedback policy.
As a result, evaluating or comparing algorithms is fundamentally indirect.

Our framework reverses the usual direction of optimal control:

> **Instead of searching for the optimal policy of a given system,
> we search for a system for which a *given* policy is optimal.**

This “converse optimality” principle enables designing environments where:

* the **optimal controller is analytically known**
* the **Bellman identity holds exactly**
* **nonlinearity and stochasticity are injected afterward**, while preserving optimality
* **difficulty is parameterized continuously**

It creates RL environments with **certified ground-truth**, ideal for robust benchmarking.

---

## **Repository Structure**

```
Code/
    ├── benchmarks/
        ├── envs/
            ├── __init__.py
            ├── lqr_env.py
            └── nudex_env.py
        ├── specs/
            ├── LQR_doubleint_hard_v1.yaml
            └── NUDEx_K1_easy_v1.yaml
        ├── __init__.py
        ├── oracles.py
        └── registry.py
    ├── cards/
        └── nudex_k1_easy_v1.md
    ├── dataset/
        ├── code/
            └── converse_four_systems_with_disturbances.py
        ├── figures/
            ├── fig_S1_COST.pdf
            ├── fig_S1_COST.png
            ├── fig_S2_COST.pdf
            ├── fig_S2_COST.png
            ├── fig_S3_COST.pdf
            ├── fig_S3_COST.png
            ├── fig_S4_COST.pdf
            └── fig_S4_COST.png
        ├── math/
            └── converse_four_systems_math.pdf
        ├── manifest.json
        └── README.md
    ├── nvdex runs/
        ├── CSVs/
            └── summary_nudex.csv
        ├── evaluation/
            ├── plotting code/
                └── plot_summary_nudex_suite.py
            └── analyze_nudex_runs.py
        ├── plotting code/
            ├── plot_summary_nudex_suite.py
            └── r
        ├── nudex_env.py
        ├── nudex_family.py
        ├── resume_all_nudex_runs.sh
        ├── run_all_nudex.sh
        ├── run_nudex_k2_parallel_resume.sh
        └── train_baselines_nudex.py
    ├── tests/
        └── test_sanity.py
    ├── yaml configs/
        ├── arm link/
            ├── algorithms.yaml
            ├── global.yaml
            ├── grid.yaml
            └── r
        ├── nvdex/
            ├── algorithms.yaml
            ├── difficulties.yaml
            ├── global.yaml
            ├── r
            └── schedules.yaml
        └── r
    ├── converse_arm_env.py
    ├── family.py
    ├── ReproducibilityChecklist (1).pdf
    ├── requirements.txt
    ├── run_all.sh
    └── train_baselines_arm_gpu.py
LICENSE
README.md
```

---

# **Benchmark Families**

This repository contains several canonical families derived through converse-optimality constructions.

All systems are **discrete-time**, **stochastic**, **control-affine**, and equipped with:

* a *closed-form quadratic value function*
* an *optimal linear state-feedback policy*
* nonlinear drifts constructed via differential geometry while maintaining full optimality

---

## **1. NVDEx System (Nonholonomic Vehicle with Dynamic Extension)**

A converse-optimal family based on a bicycle/unicycle geometry with input dynamic extension.
The system is:

* **locally open-loop unstable**
* **nonholonomic**
* **rotationally coupled through the swirl operator**
* solvable in closed form in the Quadratic–Gaussian specialization

Optimal policy and optimal value are known analytically.

### **Illustration**

<img width="1489" height="1590" src="https://github.com/user-attachments/assets/933a1ba4-3c23-4352-96d1-7d79de58f7c3">

---

## **2. Converse Arm System**

A serial (n)-link planar arm, with revolute joints and torque inputs.
This benchmark offers:

* a physically interpretable morphology
* highly coupled nonlinear dynamics
* closed-form optimal policies
* tunable difficulty via geometric parameters (swirl strength, conditioning, noise)

### **Illustration (uncontrolled vs controlled)**

![uncontrolled](https://github.com/user-attachments/assets/518b1069-a046-4b7c-9197-dbae76044f1f)
![controlled](https://github.com/user-attachments/assets/9277b4d9-df7f-4db9-97dc-728ed8ce18d4)

<img width="1218" height="432" alt="fig_optgap_heatmaps_grid (2)" src="https://github.com/user-attachments/assets/0fad03fd-02f1-48f7-8187-20ae695d199a" />
<img width="3904" height="1439" alt="image" src="https://github.com/user-attachments/assets/93a68d57-67b6-4384-b275-6dd5c560f767" />


---

# **Features**

### **Ground-Truth Evaluation**

Each benchmark exposes:

* `oracle.action(s)` — exact optimal action
* `oracle.value(s)` — exact value function
* `oracle.q_value(s, a)` — exact Q-function

Allowing measurement of:

* optimality gaps
* exact regret
* Bellman residuals
* finite-horizon deviation from optimality

---

### **Tunable Difficulty**

Every benchmark family supports continuous difficulty control through:

* noise covariance
* drift conditioning
* swirl strength
* dynamic extension order
* cost geometry

---

### **Reproducible Stochasticity**

Benchmarks use:

* deterministic system generation (via seeds)
* stochastic rollouts (Gaussian noise)
* stable YAML specifications (`benchmarks/specs/*.yaml`)

---

### **Unified Registry**

Benchmarks can be instantiated from a single name:

```python
from Code.benchmarks import registry
env, oracle = registry.make("NUDEx_K1_easy_v1")
```

---

# **Quick Start**

## **Installation**

```
git clone https://github.com/converseoptimality/RL-Benchmarking.git
cd RL-Benchmarking/Code
pip install -r requirements.txt
```

---

## **Using a Benchmark**

```python
from Code.benchmarks import registry

env, oracle = registry.make("NUDEx_K1_easy_v1")

obs = env.reset()

# Evaluate optimal policy
a_star = oracle.action(obs)
obs, reward, done, info = env.step(a_star)
```

---

## **Running Baseline Training**

The repository includes an example GPU training script:

```
python Code/train_baselines_arm_gpu.py
```

and a convenience launcher:

```
bash Code/run_all.sh
```
---

# **Reproducing the Paper**

This section describes how to reproduce **all experiments, figures, and numerical results** from the paper.
Experiments are divided into two benchmark families:

1. **Converse arm system**
2. **NVDEx system** (Nonholonomic Vehicle with Dynamic Extension)

Both pipelines rely only on components included in the repository and require no external data.

---

## **Prerequisites**

Make sure the required packages are installed (as described higher).

Ensure the repository root is on the Python path:

```bash
export PYTHONPATH=$PWD
```

GPU is recommended but not required for small-scale runs.

---

# **Reproducing the arm experiments**

The arm benchmark experiments include:

* Optimal vs uncontrolled rollouts
* Optimality gap evaluations
* Grid-based heatmaps
* Difficulty-ladder trajectories
* Figures used in the paper

All arm benchmark configuration files are located in:

```
Code/yaml configs/arm link/
```

### **Run arm training**

Use the main training script:

```bash
python Code/train_baselines_arm_gpu.py
```

This script reads the following configuration files:

* `yaml configs/arm link/algorithms.yaml` — RL hyperparameters
* `yaml configs/arm link/global.yaml` — global experimental settings
* `yaml configs/arm link/grid.yaml` — grid rollout specification

The script generates:

```
arm_results/
    ├── logs/                # raw training logs
    ├── models/              # trained policies
    ├── eval/                # rollout CSVs
    └── figs/                # learning curves, heatmaps, ideal vs uncontrolled
```

*(If directory names differ slightly on your local version, adjust accordingly.)*

---

### **Reproduce arm benchmark evaluation/plots**

After training, generate all arm figures using:

```bash
bash Code/run_all.sh
```

This script performs:

* RL policy evaluation
* Optimality gap computation
* Heatmap generation
* All plots used in the paper

This includes:

* Controlled vs uncontrolled rollouts
* State heatmaps
* Optimality gap plots
* Difficulty ladder rollouts

---

### **Reproduce arm and mathematical results**

Supplementary ARM figures (S1–S4) can be regenerated via:

```bash
python Code/dataset/code/converse_four_systems_with_disturbances.py
```

Figures are written to:

```
Code/dataset/figures/
```

---

# **Reproducing the NUDEx Experiments**

NVDEx (NUDEx) experiments include:

* Optimal vs uncontrolled trajectories
* Difficulty ladder analysis
* Heatmaps and cost distributions
* Performance summaries across initial conditions
* All supplementary NVDEx figures

NUDEx-specific configuration files are located in:

```
Code/yaml configs/nvdex/
```

### **Run training for NUDEx**

The core training script:

```bash
python Code/nvdex runs/train_baselines_nudex.py
```

Reads:

* `yaml configs/nvdex/algorithms.yaml`
* `yaml configs/nvdex/global.yaml`
* `yaml configs/nvdex/difficulties.yaml`
* `yaml configs/nvdex/schedules.yaml`

and produces:

```
nvdex runs/
    ├── CSVs/
    └── evaluation/
```

---

### Run the entire pipeline in parallel:

```bash
bash Code/nvdex runs/run_all_nudex.sh
```

---

### **Generate NUDEx evaluation figures**

After training is complete, run:

```bash
python Code/nvdex runs/evaluation/analyze_nudex_runs.py
```

and generate summary plots using:

```bash
python Code/nvdex runs/evaluation/plotting code/plot_summary_nudex_suite.py
```

This produces:

```
nvdex runs/CSVs/summary_nudex.csv
```

This directory contains:

* Difficulty ladder performance
* State and control trajectories
* Heatmaps
* All plots used in paper figures

---

# **Verification**

Correctness can be checked via:

```bash
pytest Code/tests/test_sanity.py
```

This ensures:

* correct environment construction
* correct oracle (`π*`, `V*`) evaluations
* consistent interface with RL algorithms

---

# **Notes**

* All configurations are version-controlled via YAML files.
* All figures from the paper correspond to the outputs of the scripts above.
* No external assets are required.

# **Results**

### **NVDEx performance across initializations & difficulty ladder**

<img width="1489" height="1589" src="https://github.com/user-attachments/assets/984c3668-c465-48db-ab64-5a614825ae01" />

### **Arm system: optimal vs uncontrolled**

<img width="1044" height="759" src="https://github.com/user-attachments/assets/2087eeee-0373-4d55-ac26-494f148da720" />

### Additional data from experiments

<img width="892" height="397" alt="Без имени" src="https://github.com/user-attachments/assets/05a2c2a1-961d-4ee5-9c08-5cf401cfaacd" />

---

# **Dataset**

The `Code/dataset/` directory provides:

* benchmark metadata & manifest
* four canonical systems (math + code)
* figures used in supplementary material
* scripts generating QG-specialized dynamics

This dataset corresponds to the figures provided in the paper.

---

# **Testing**

Basic sanity tests validate that each benchmark:

* initializes without error
* exposes `reset`, `step`, and `oracle` interface
* preserves shape and dtype guarantees

Run tests via:

```
pytest Code/tests/
```

---

# **Citation**

If you use this repository or benchmark suite in your research, please cite the paper:

```
@article{<to-be-added>,
  title={Benchmarking RL via Converse Optimality: Generating Systems with Known Optimal Policies},
  author={...},
  year={2025},
  journal={...},
}
```

A final BibTeX entry will be added upon publication.

---
