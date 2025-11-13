import numpy as np
from gymnasium import Env, spaces

class CustomGymEnv(Env):
    """
    Gym environment wrapping any GenericODE, with cost-based reward.

    Reward at each step is:
        r = - cost_fun(x, u) * dt

    If no cost_fun is provided, falls back to -||x||^2 - ||u||^2.
    """
    def __init__(self,
                 system,
                 cost_fun=None,
                 dt: float = 0.01,
                 max_time_steps: int = 700):
        super().__init__()
        self.system          = system
        self.cost_fun        = cost_fun
        self.dt              = dt
        self.max_time_steps  = max_time_steps
        self.current_time_step = 0

        # initial state = vector of 1.1’s by default
        self._init_val = 1.1
        self.state_scale = 1.0

        # dimensions
        self.dim         = system._dim_state
        self.dim_inputs  = system._dim_inputs

        # initialize state
        self.state = np.full((self.dim,), self._init_val, dtype=np.float32)

        # observation & action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-100, high=100,
            shape=(self.dim_inputs,), dtype=np.float32
        )

    def step(self, action):
        # 1) advance the dynamics
        self.current_time_step += 1
        scaled = self.state / self.state_scale
        sdot   = self.system.compute_state_dynamics(scaled, action)
        if np.isnan(sdot).any():
            raise ValueError("NaN detected in dynamics/state")
        # Euler integrate
        self.state = (self.state + sdot.T * self.dt).astype(np.float32)
        self.state = self.state.flatten() / self.state_scale

        # # 2) compute reward
        # if self.cost_fun is not None:
        c_val  = float(self.cost_fun(self.state, action))
        reward = -c_val * self.dt
        #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # else:
        #     reward = -np.sum(self.state**2) - np.sum(action**2)

        # 3) termination / truncation
        terminated = np.linalg.norm(self.state) < 1e-4
        truncated  = (self.current_time_step >= self.max_time_steps)
        done       = terminated or truncated

        info = {}
        return self.state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.current_time_step = 0
        self.state = np.full((self.dim,), self._init_val, dtype=np.float32)
        return self.state, {}

    def render(self):
        print(f"Current state: {self.state}")
