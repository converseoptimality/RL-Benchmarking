import numpy as np, gymnasium as gym
from gymnasium.spaces import Box
from typing import Optional, Dict, Any
from nudex_family import build_family

def _stage_cost(fam, s, a): return float(s.T @ fam.Q @ s + a.T @ fam.R @ a)
def _true_value(fam, s):    return float(s.T @ fam.P @ s + fam.c_cost)

def _indices_module(i: int, r_v: int, r_w: int, n_mod: int):
    base = i * n_mod
    return {
        "x": base + 0,
        "y": base + 1,
        "phi": base + 2,
        "v0": base + 3,
        "w0": base + 3 + r_v,
    }

class NUDExEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self,
                 # structure
                 K: int = 1, r_v: int = 1, r_w: int = 1, tau: float = 0.1,
                 # discount/control
                 gamma: float = 0.99, p: float = 1.0,
                 # spectra
                 weight_pose: float = 2.0, weight_v: float = 1.0, weight_w: float = 1.0,
                 r_u_v: float = 1.0, r_u_w: float = 1.0,
                 # noise
                 sigma_pose: float = 1e-6, sigma_v: float = 5e-4, sigma_w: float = 5e-4,
                 # field
                 alpha: float = 0.08, kappa: float = 0.3,
                 # instability target
                 rho_target: float = 1.005, allow_R_rescale: bool = False, # rescale to make control cheaper if needed
                 R_scale: float = 0.6, max_rescale_iter: int = 10,
                 # gym opts
                 horizon: int = 700, act_limit: float = 4.0,
                 obs_clip: Optional[float] = None,
                 seed: int = 7,
                 render_mode: Optional[str] = None,
                 # fairness
                 fixed_init: bool = True,
                 reset_seed_per_episode: bool = True):
        super().__init__()

        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.p = float(p)
        self.obs_clip = obs_clip
        self.render_mode = render_mode

        # build one family
        self.fam = build_family(
            K=K, r_v=r_v, r_w=r_w, tau=tau, gamma=gamma, p=p,
            weight_pose=weight_pose, weight_v=weight_v, weight_w=weight_w,
            r_u_v=r_u_v, r_u_w=r_u_w,
            sigma_pose=sigma_pose, sigma_v=sigma_v, sigma_w=sigma_w,
            alpha=alpha, kappa=kappa,
            rho_target=rho_target, allow_R_rescale=allow_R_rescale,
            R_scale=R_scale, max_rescale_iter=max_rescale_iter, seed=seed
        )

        # dims
        self.K, self.r_v, self.r_w = K, r_v, r_w
        self.n_mod = 3 + r_v + r_w
        self.n = self.fam.state_dim
        self.m = self.fam.action_dim

        self.action_space = Box(low=-act_limit, high=act_limit,
                                shape=(self.m,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=(self.n,), dtype=np.float32)

        # fairness bookkeeping
        self.base_seed = int(seed)
        self.reset_seed_per_episode = bool(reset_seed_per_episode)
        self.fixed_init = bool(fixed_init)
        self._s0_fixed = None
        self.ep_idx = 0

        self.t = 0
        self.s = np.zeros(self.n, dtype=float)

        self.rng = np.random.default_rng(self.base_seed)
        self.fam.rng = np.random.default_rng(self.base_seed + 1)

    # ---- Gym API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.base_seed = int(seed)
            self.ep_idx = 0

        if self.reset_seed_per_episode:
            epi_seed = self.base_seed + 1_000_001 * self.ep_idx
            self.rng = np.random.default_rng(epi_seed)
            self.fam.rng = np.random.default_rng(epi_seed + 7)

        if self.fixed_init and (self._s0_fixed is not None):
            s0 = self._s0_fixed.copy()
        else:
            s0 = np.zeros(self.n, dtype=float)
            # small offsets near origin for each module
            for i in range(self.K):
                idx = _indices_module(i, self.r_v, self.r_w, self.n_mod)
                th = self.rng.uniform(-np.pi, np.pi)
                r = 0.12 * (0.6 + 0.4 * self.rng.random())
                s0[idx["x"]]   = r * np.cos(th)
                s0[idx["y"]]   = r * np.sin(th)
                s0[idx["phi"]] = self.rng.normal(scale=0.06)
                s0[idx["v0"]]  = self.rng.normal(scale=0.05)
                s0[idx["w0"]]  = self.rng.normal(scale=0.05)
        if self.fixed_init and (self._s0_fixed is None):
            self._s0_fixed = s0.copy()

        self.s = s0
        self.t = 0

        obs = self._obs(self.s)
        info = {
            "p": self.p, "gamma": self.gamma,
            "true_value": _true_value(self.fam, self.s),
            "c_cost": self.fam.c_cost,
            "ep_idx": self.ep_idx
        }
        self.ep_idx += 0
        return obs, info

    def step(self, a: np.ndarray):
        a = np.asarray(a, dtype=float)
        a = np.clip(a, self.action_space.low, self.action_space.high)
        s_next = self.fam.step(self.s, a, self.p, stochastic=True)

        if not np.all(np.isfinite(s_next)) or np.linalg.norm(s_next) > 1e6:
            # keep things finite, end episode with a finite penalty
            s_next = np.nan_to_num(np.clip(s_next, -1e6, 1e6))
            self.s = s_next
            reward = -1e6  # finite sentinel; keep it modest so advantages don't blow up
            terminated, truncated = False, True
            info = {"true_value": float(s_next.T @ self.fam.P @ s_next + self.fam.c_cost)}
            return self._obs(self.s), reward, terminated, truncated, info
        
        cost = _stage_cost(self.fam, self.s, a)
        reward = -cost
        self.s = s_next
        self.t += 1
        terminated = False
        truncated = self.t >= self.horizon
        info = {"true_value": _true_value(self.fam, self.s)}
        return self._obs(self.s), reward, terminated, truncated, info

    def render(self):
        assert self.render_mode == "rgb_array", "Only rgb_array is supported"
        # minimalist pose scatter for module 0
        import matplotlib.pyplot as plt
        from io import BytesIO
        idx = _indices_module(0, self.r_v, self.r_w, self.n_mod)
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.scatter([self.s[idx["x"]]], [self.s[idx["y"]]], s=40, c="tab:blue")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        plt.close(fig)
        return buf

    # ---- helpers ----
    def _obs(self, s):
        o = s.astype(np.float32)
        if self.obs_clip is not None:
            np.clip(o, -self.obs_clip, self.obs_clip, out=o)
        return o

# --- register ---
from gymnasium.envs.registration import register
register(id="NUDEx-v0", entry_point="nudex_env:NUDExEnv")