import numpy as np
from gymnasium import spaces, Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.spaces import Box

class CustomGymEnv(Env):
    def __init__(self, system, max_time_steps=100):
        """
        A custom Gym environment for SB3 training.
        Args:
            system (BenchmarkSysType1): A system that defines the dynamics.
            max_time_steps (int): Maximum steps per episode.
        """
        super().__init__()
        self.system = system
        self.dim = system._dim_state
        self.dim_inputs = system._dim_inputs
        self.max_time_steps = max_time_steps
        self.current_time_step = 0
        self.done = False

        # # Define action and observation spaces
        # self.action_space = spaces.Box(
        #     low=np.array([bound[0] for bound in self.system._action_bounds], dtype=np.float32),
        #     high=np.array([bound[1] for bound in self.system._action_bounds], dtype=np.float32),
        #     dtype=np.float32
        # )
        # self.min_action = np.full(self.dim_inputs, -10, dtype=np.float32)
        # self.max_action = np.full(self.dim_inputs, 10, dtype=np.float32)
        # self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=(self.dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10,
            high=10,
            shape=(self.dim_inputs,),
            dtype=np.float32
        )


        # Initialize the state
        #self.state = np.random.uniform(-2.00, -2.0, self.dim).astype(np.float32)
        #self.state = np.array([-4.0, -5.0, -6.0], dtype=np.float32)
        self.state = np.array([-1.5, -1.5, -1.5], dtype=np.float32) # -4, -5, -6, it was 1.5 
        # self.state = np.array(self.dim, -1.5, dtype=np.float32) # -4, -5, -6

        self.state_scale = 1  # Scaling factor

    def step(self, action):
        """
        The step function is a function compatible with stb3 and affects what happens in each timestep during the training.
        Args:
            action: the action predicted by the controller.
        """
        self.current_time_step += 1
        scaled_state = self.state / self.state_scale
        #action= action * 10
        sdot = self.system.compute_state_dynamics(scaled_state, action)

        # Check for numerical instabilities
        if np.isnan(sdot).any() or np.isnan(scaled_state).any():
            raise ValueError("NaN detected in dynamics or state!")

        # Update state and apply scaling
        self.state = (self.state + sdot.T * 0.01).astype(np.float32) # 0.5 or 0.1 for 3s,1a || 0.1 for the 5s 3a
        self.state = self.state / self.state_scale

        # Define reward and termination criteria
        #reward = - np.linalg.norm(self.state) #-np.dot(self.state, self.state) - np.dot(action, action)
        reward = -np.sum(np.square(self.state)) - np.sum(np.square(action))
        reward = np.clip(reward, -3000, 3000)
        self.done = np.linalg.norm(reward) <  0.01 or self.current_time_step >= self.max_time_steps

        truncated = self.current_time_step >= self.max_time_steps
        info = {}
        if self.current_time_step % 10 == 0:
            print("obs:",self.state," - reward: ", reward, " - action: ", action, " - current_timestep: ", self.current_time_step)

        return self.state, reward, self.done, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        #self.state = np.random.uniform(-2.0, -2.0, self.dim).astype(np.float32)
        self.state = np.array([-1.5, -1.5, -1.5], dtype=np.float32) # -4, -5, -6, it was 1.5 
        #self.state = np.array(self.dim, -1.5, dtype=np.float32) # -4, -5, -6

        self.current_time_step = 0
        self.done = False
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        print(f"Current state: {self.state}")
