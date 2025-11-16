"""
Minimal oracle dispatch. Real oracles should be plugged in per family.
"""
from dataclasses import dataclass
import numpy as np
from typing import Callable
from .registry import load_spec

@dataclass
class Oracle:
    action_fn: Callable[[np.ndarray], np.ndarray]
    def action(self, s: np.ndarray) -> np.ndarray:
        return self.action_fn(s)

def _lqr_oracle(spec: dict) -> Oracle:
    import numpy as np
    from scipy.linalg import solve_discrete_are
    A = np.array(spec["params"]["A"], dtype=float)
    B = np.array(spec["params"]["B"], dtype=float)
    Q = np.array(spec["params"]["Q"], dtype=float)
    R = np.array(spec["params"]["R"], dtype=float)
    gamma = float(spec["gamma"])
    # Discounted DARE: solve ARE on (sqrt(gamma)A, sqrt(gamma)B)
    Ag = np.sqrt(gamma) * A
    Bg = np.sqrt(gamma) * B
    P = solve_discrete_are(Ag, Bg, Q, R)
    K = np.linalg.inv(R + gamma * B.T @ P @ B) @ (gamma * B.T @ P @ A)
    def act(s: np.ndarray) -> np.ndarray:
        return (-K @ s).astype(np.float32).reshape(-1)
    return Oracle(act)

def get_oracle(spec_id: str) -> Oracle:
    spec = load_spec(spec_id)
    fam = spec["family"].lower()
    if fam == "lqr":
        return _lqr_oracle(spec)
    # Stubs for other families—replace with your analytic oracles.
    return Oracle(lambda s: np.zeros(spec["action_dim"], dtype=np.float32))
