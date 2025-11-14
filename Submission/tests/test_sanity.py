import pytest, numpy as np
from benchmarks.registry import register_all
import gymnasium as gym

def test_lqr_rolls():
    register_all()
    env = gym.make("LQR-doubleint-hard-v0")
    s,_ = env.reset(seed=0)
    assert s.shape == (2,)
    a = np.zeros(1, dtype=np.float32)
    s2,r,done,tr,info = env.step(a)
    assert s2.shape == (2,)
    assert isinstance(r, float)
