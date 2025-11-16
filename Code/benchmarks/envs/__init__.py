import gymnasium as gym
from .lqr_env import LQREnv
from .nudex_env import NUDExEnv

def make_from_spec(spec: dict) -> gym.Env:
    fam = spec["family"].lower()
    if fam == "lqr":
        return LQREnv(spec)
    if fam == "nudex":
        return NUDExEnv(spec)  # requires your system builder
    raise ValueError(f"Unknown family: {fam}")
