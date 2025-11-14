# Converse Benchmark (Blinded Skeleton)

> **Double‑blind, anonymized repository skeleton.**  
> This repo contains a minimal, runnable scaffold for the converse‑optimal / QG benchmark dataset and envs.
> Replace placeholders with your actual implementations after blinding.

## What’s here
- **`benchmarks/`** — package with `envs/`, `specs/`, `registry.py`, and stubbed `oracles.py`.
- **`cards/`** — dataset cards (Markdown).
- **`tests/`** — smoke tests / quality gates.
- **`ANONYMIZATION.md`** — blinding checklist + how to use Anonymous GitHub (4open).
- **`requirements.txt`** — Python deps (Gymnasium, NumPy, SciPy, PyYAML).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Register and run an example env (LQR double integrator)
python -c "from benchmarks.registry import register_all;register_all();import gymnasium as gym;env=gym.make('LQR-doubleint-hard-v0');s,_=env.reset(seed=0);import numpy as np;done=False;ret=0;import benchmarks.oracles as O;ora=O.get_oracle('LQR/doubleint/hard/v1'); import math; while not done: a=ora.action(s); s,r,done,tr,info=env.step(a); ret+=r; print('Return:',ret)"
```

> For the **NUDEx / ConverseArm** families, this skeleton provides specs, env stubs, and integration points — drop in your system builders and oracles.

## Colab (optional)
If you maintain a Colab for reviewers, put the shared link here (anonymized).

## After acceptance
Replace anonymous links with the canonical repository and fill in the license/authors.
