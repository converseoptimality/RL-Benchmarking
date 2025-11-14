# converse_arm_env.py
import numpy as np, gymnasium as gym
from gymnasium.spaces import Box
from typing import Optional, Dict, Any
from family import build_family

def _stage_cost(fam, s, a): return float(s.T @ fam.Q @ s + a.T @ fam.R @ a)
def _true_value(fam, s):    return float(s.T @ fam.P @ s + fam.c_cost)

class ConverseArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self,
                 n_links: int = 6,
                 gamma: float = 0.99,
                 r_u: float = 2.0,
                 tau: float = 0.05,
                 sigma_theta: float = 0.0,
                 sigma_omega: float = 1e-3,
                 p: float = 0.8,
                 horizon: int = 512,
                 act_limit: float = 2.0,
                 obs_clip: Optional[float] = None,
                 seed: int = 7,
                 render_mode: Optional[str] = None,
                 # appearance / dynamics “flavor”
                 grav0: float = 0.8,
                 swirl_gain: float = 0.5,
                 grav_always_on: bool = True,
                 # NEW: fairness controls
                 fixed_init: bool = True,               # if True: same s0 every episode
                 reset_seed_per_episode: bool = True     # if True: reseed rngs at EVERY reset
                 ):
        super().__init__()
        self.gamma = gamma
        self.horizon = int(horizon)
        self.p = float(p)
        self.obs_clip = obs_clip
        self.render_mode = render_mode

        # Build ONE family (system) deterministically from the provided seed
        self.fam = build_family(n=n_links, gamma=gamma, r_u=r_u, tau=tau,
                                sigma_theta=sigma_theta, sigma_omega=sigma_omega,
                                seed=seed, grav0=grav0, swirl_gain=swirl_gain,
                                grav_always_on=grav_always_on)

        self.n = self.fam.n
        self.d = 2*self.n
        self.act_limit = float(act_limit)

        self.action_space = Box(low=-self.act_limit, high=self.act_limit,
                                shape=(self.n,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=(self.d,), dtype=np.float32)

        # --- NEW: fairness bookkeeping ---
        self.base_seed = int(seed)                  # defines family + episode sequence
        self.reset_seed_per_episode = bool(reset_seed_per_episode)
        self.fixed_init = bool(fixed_init)
        self._s0_fixed = None
        self.ep_idx = 0  # counts completed resets so far

        # state
        self.t = 0
        self.s = np.zeros(self.d, dtype=float)

        # RNGs (will be reseeded in reset)
        self.rng = np.random.default_rng(self.base_seed)
        self.fam.rng = np.random.default_rng(self.base_seed + 1)

    # ---- Gym API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # If Gym gives a new seed, use it as the new base for EVERYTHING
        if seed is not None:
            self.base_seed = int(seed)
            self.ep_idx = 0

        # Deterministic reseeding per episode (critical for fairness)
        if self.reset_seed_per_episode:
            # Use a large stride to avoid overlaps
            epi_seed = self.base_seed + 1000003 * self.ep_idx
            self.rng = np.random.default_rng(epi_seed)
            self.fam.rng = np.random.default_rng(epi_seed + 7)

        # Sample (or reuse) initial state
        if self.fixed_init and (self._s0_fixed is not None):
            s0 = self._s0_fixed.copy()
        else:
            th = self.rng.uniform(-2.8, 2.8, size=self.n)
            om = self.rng.uniform(-1.2, 1.2, size=self.n)
            s0 = np.empty(self.d); s0[0::2] = th; s0[1::2] = om
            if self.fixed_init and (self._s0_fixed is None):
                self._s0_fixed = s0.copy()

        self.s = s0
        self.t = 0

        obs = self._obs(self.s)
        #print(obs)
        info = {
            "p": self.p, "gamma": self.gamma,
            "tau": float(np.trace(self.fam.Q)/np.trace(self.fam.P)),
            "true_value": _true_value(self.fam, self.s),
            "c_cost": self.fam.c_cost,
            "ep_idx": self.ep_idx
        }

        # increment AFTER creating s0 so episode 0 uses ep_idx=0 seeds
        self.ep_idx += 1
        return obs, info

    def step(self, a: np.ndarray):
        a = np.clip(a, -self.act_limit, self.act_limit).astype(float)
        s_next = self.fam.step(self.s, a, self.p, stochastic=True)
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
        return self._draw_frame(self.s)

    # ---- helpers ----
    def _obs(self, s):
        o = s.astype(np.float32)
        if self.obs_clip is not None:
            np.clip(o, -self.obs_clip, self.obs_clip, out=o)
        return o

    # simple renderer (single gravity arrow + torque hints); fast & headless
    def _draw_frame(self, s):
        import matplotlib.pyplot as plt
        from io import BytesIO

        def link_positions(theta, L=1.0):
            x=[0.0]; y=[0.0]; ang=0.0
            for th in theta:
                ang += th
                x.append(x[-1] + L*np.cos(ang))
                y.append(y[-1] + L*np.sin(ang))
            return np.array(x), np.array(y)

        theta = s[0::2]
        X, Y = link_positions(theta, L=1.0)
        fig, ax = plt.subplots(figsize=(4,4))
        ax.set_aspect('equal', 'box')
        pad = self.n*0.2
        ax.set_xlim(-self.n-pad, self.n+pad); ax.set_ylim(-self.n-pad, self.n+pad)
        ax.grid(True, alpha=0.2)
        ax.plot(X, Y, '-o', lw=3, ms=5, color='tab:blue')
        ax.plot(0,0,'ko', ms=5); ax.text(0.1,-0.15,"fixed base", fontsize=8)

        # gravity arrow (bottom-left)
        gx, gy = -self.n-pad*0.8, -self.n-pad*0.2
        ax.arrow(gx, gy, 0, -0.8, width=0.02, head_width=0.15, head_length=0.15, color='tab:gray')
        ax.text(gx+0.1, gy-1.0, "gravity", fontsize=9, color='tab:gray')

        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h,w,4)[..., :3]
        plt.close(fig)
        return buf


# --- at very end of converse_arm_env.py ---
from gymnasium.envs.registration import register

register(
    id="ConverseArm-v0",
    entry_point="converse_arm_env:ConverseArmEnv",
)
