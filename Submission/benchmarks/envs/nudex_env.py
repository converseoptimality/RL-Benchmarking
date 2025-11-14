import gymnasium as gym
import numpy as np

class NUDExEnv(gym.Env):
    """
    Stub env wiring consistent with your system builder.
    Replace `build_nudex_system_unstable` with your implementation.
    """
    metadata = {"render_modes": []}
    def __init__(self, spec: dict):
        super().__init__()
        self.spec = spec
        p = spec["params"]
        n = spec["state_dim"]
        m = spec["action_dim"]
        # ---- plug your core system here ----
        # from .nudex_core import build_nudex_system_unstable
        # self.sys = build_nudex_system_unstable(...)
        self.sys = None  # placeholder
        # ------------------------------------
        act_limit = spec.get("constraints", {}).get("act_limit", 2.0)
        self.action_space = gym.spaces.Box(low=-act_limit, high=act_limit, shape=(m,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32)
        self._rng = np.random.default_rng(spec.get("init", {}).get("crn_seed_base", 7))
        self._noise_seed = spec.get("init", {}).get("crn_seed_base", 7)
        self._crn_mode = (spec.get("init", {}).get("mode", "CRN") == "CRN")
        self.state = None
        self.t = 0

    def reseed_noise(self):
        self._noise_seed += 1
        self._rng = np.random.default_rng(self._noise_seed)

    def sample_s0(self):
        # Example ring sampler; adapt to your indices_module mapping
        n = self.spec["state_dim"]
        s0 = np.zeros(n)
        # TODO: fill using your indices / per-vehicle ring sampler
        return s0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._crn_mode:
            self.reseed_noise()
        self.state = self.sample_s0()
        self.t = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        a = np.clip(action, self.action_space.low, self.action_space.high)
        # w = self.sys.sample_noise(self._rng)   # when wired
        # s_next = self.sys.step(self.state, a, w)
        s_next = self.state  # placeholder to keep skeleton runnable
        # cost = self.sys.cost(self.state, a)
        cost = float(np.sum(self.state**2) + np.sum(a**2))  # placeholder cost
        reward = -cost
        self.state = s_next
        self.t += 1
        done = (self.t >= self.spec["horizon"])
        truncated = False
        return self.state.astype(np.float32), reward, done, truncated, {}
