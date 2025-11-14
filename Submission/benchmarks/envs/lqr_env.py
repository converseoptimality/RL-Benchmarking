import gymnasium as gym
import numpy as np

class LQREnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, spec: dict):
        super().__init__()
        self.spec_dict = spec
        p = spec["params"]
        self.A = np.array(p["A"], dtype=float)
        self.B = np.array(p["B"], dtype=float)
        self.Q = np.array(p["Q"], dtype=float)
        self.R = np.array(p["R"], dtype=float)
        self.gamma = float(spec["gamma"])
        self.dt = float(spec["dt"])
        self.horizon = int(spec["horizon"])
        self.n = int(spec["state_dim"])
        self.m = int(spec["action_dim"])
        self.process_std = float(spec.get("noise", {}).get("process_std", 0.0))
        # Spaces
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.m,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n,), dtype=np.float32)
        # RNG for CRN
        base = int(spec.get("init", {}).get("crn_seed_base", 0))
        self._rng = np.random.default_rng(base)
        self._noise_seed = base
        self._crn_mode = (spec.get("init", {}).get("mode", "CRN") == "CRN")
        self.state = None
        self.t = 0

    def reseed_noise(self):
        self._noise_seed += 1
        self._rng = np.random.default_rng(self._noise_seed)

    def sample_s0(self):
        sam = self.spec_dict.get("init", {}).get("sampler", {})
        mean = np.array(sam.get("mean", [0.0] * self.n), dtype=float)
        cov = np.array(sam.get("cov", np.eye(self.n)), dtype=float)
        return self._rng.multivariate_normal(mean, cov)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._crn_mode:
            self.reseed_noise()
        self.state = self.sample_s0()
        self.t = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        a = np.asarray(action, dtype=float).reshape(self.m)
        # Clip (optional) — here unconstrained
        w = self._rng.normal(0.0, self.process_std, size=self.n)
        s_next = self.A @ self.state + self.B @ a + w
        # Cost and reward
        cost = float(self.state.T @ self.Q @ self.state + a.T @ self.R @ a)
        reward = -cost
        self.state = s_next
        self.t += 1
        done = (self.t >= self.horizon)
        truncated = False
        return self.state.astype(np.float32), reward, done, truncated, {}
