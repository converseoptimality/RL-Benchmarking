# **Benchmarking RL via Converse Optimality**

*Official Repository — [https://github.com/converseoptimality/RL-Benchmarking](https://github.com/converseoptimality/RL-Benchmarking)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## **Overview**

This repository contains the official implementation accompanying
**“Benchmarking RL via Converse Optimality: Generating Systems with Known Optimal Policies”**
(submitted for peer review).

The project provides a **scientifically rigorous benchmarking framework** for reinforcement learning, built around *converse-optimality constructions* that generate dynamical systems with:

* **provably known optimal policies** (π^*)
* **closed-form optimal value functions** (V^*)
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
RL-Benchmarking/
│
├── Submission/
│   ├── benchmarks/                 # Benchmark families + registry
│   │   ├── envs/
│   │   │   ├── lqr_env.py          # LQR-type baseline environment
│   │   │   └── nudex_env.py        # NUDEx nonlinear vehicle benchmark
│   │   ├── specs/                  # YAML benchmark specifications
│   │   ├── oracles.py              # Ground-truth (π*, V*) oracles
│   │   └── registry.py             # Benchmark registration utilities
│   │
│   ├── cards/                      # Benchmark cards (metadata + plots)
│   │   └── nudex_k1_easy_v1.md
│   │
│   ├── dataset/                    # Published benchmark dataset (Fig S1–S4)
│   │   ├── code/
│   │   ├── figures/
│   │   ├── math/
│   │   └── manifest.json
│   │
│   ├── tests/
│   │   └── test_sanity.py          # Basic environment initialization tests
│   │
│   ├── converse_arm_env.py         # Converse-optimal arm dynamics
│   ├── family.py                   # Benchmark family construction utilities
│   ├── ReproducibilityChecklist.pdf
│   ├── run_all.sh                  # Example workflow script
│   └── train_baselines_arm_gpu.py  # Training script (GPU-backed)
│
├── LICENSE
└── README.md
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

### Placeholder for additional NVDEx figures

*(e.g., drift field visualization, trajectories for difficulty ladder)*

```
[Insert additional NVDEx figure here]
```

---

## **2. Converse Arm System**

A serial (n)-link planar arm, with revolute joints and torque inputs.
This benchmark offers:

* a physically interpretable morphology
* highly coupled nonlinear dynamics
* closed-form optimal policies
* tunable difficulty via geometric parameters (swirl strength, conditioning, noise)

### **Illustration (controlled vs uncontrolled)**

![uncontrolled](https://github.com/user-attachments/assets/518b1069-a046-4b7c-9197-dbae76044f1f)
![controlled](https://github.com/user-attachments/assets/9277b4d9-df7f-4db9-97dc-728ed8ce18d4)

### Placeholder for additional arm figures

```
[Insert arm benchmark trajectories / value heatmaps here]
```

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
from Submission.benchmarks import registry
env, oracle = registry.make("NUDEx_K1_easy_v1")
```

---

# **Quick Start**

## **Installation**

```
git clone https://github.com/converseoptimality/RL-Benchmarking.git
cd RL-Benchmarking
pip install -r requirements.txt
```

---

## **Using a Benchmark**

```python
from Submission.benchmarks import registry

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
python Submission/train_baselines_arm_gpu.py
```

and a convenience launcher:

```
bash Submission/run_all.sh
```

More scripts can be added to `Submission/experiments/` later.

### Placeholder for additional training instructions

```
[Insert training workflow for PPO/SAC/TD3 etc.]
```

---

# **Results**

### **NVDEx performance across initializations & difficulty ladder**

<img width="1489" height="1589" src="https://github.com/user-attachments/assets/984c3668-c465-48db-ab64-5a614825ae01" />

### **Arm system: optimal vs uncontrolled**

<img width="1044" height="759" src="https://github.com/user-attachments/assets/2087eeee-0373-4d55-ac26-494f148da720" />

### Placeholder for additional experiment plots

```
[Insert learning curves, regret curves, Bellman error plots]
```

---

# **Dataset**

The `Submission/dataset/` directory provides:

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
pytest Submission/tests/
```

---

# **Planned Extensions**

Future updates (post-review) will include:

* full training suite (PPO, A2C, SAC, TD3)
* multi-agent systems
* Gymnasium API wrappers
* hyperparameter configurations
* notebook tutorials
* statistical analysis utilities

Placeholders are already included in the README to incorporate these seamlessly.

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
