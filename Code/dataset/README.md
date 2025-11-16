# datastet ✨

**Ground‑truth RL mini‑suite (Converse‑Optimal, QG)**  
Four physically‑meaningful systems, analytic optimal policy/value, and internal OU/ripple disturbances — packaged for quick plots & reproducible tests.

## What’s inside
- **figures/** → 4× *discounted cumulative cost* (optimal vs no‑control, CRN) + overlay variants (png+pdf)
- **code/** → ready‑to‑run scripts to regenerate all figures
- **math/** → LaTeX with full equations (H_γ, drift, policy, offset) for all systems
- **manifest.json** → quick index of files

## Quickstart
```bash
# Cost‑vs‑time (optimal vs no‑control, CRN), saves to figures/
python3 code/converse_four_systems_with_disturbances_COST.py

# Overlay (no‑control vs optimal) with internal disturbances
python3 code/converse_four_systems_with_disturbances.py

# (Optional) Original four‑systems (no internal disturbances)
python3 code/converse_four_systems.py

# (Optional) Single manipulator demo
python3 code/converse_manipulator_demo.py
```

## Cite
If this helps your research, please cite the stochastic *converse optimality* framework used here (QG specialization, discounted Schur metric, metric‑normalized drift, analytic policy).

---
*Made with ❤️ for clean, certified RL baselines.  Stay optimal.*
