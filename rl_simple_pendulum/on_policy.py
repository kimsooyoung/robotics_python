import gymnasium as gym
import numpy as np
import torch
import torch as th
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "context": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                ),
                "score": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        return {
            "context": np.array([1.0, 2.0], dtype=np.float32),
            "score": np.array([0.0], dtype=np.float32),
        }, {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return (
            {
                "context": np.array([1.0, 2.0], dtype=np.float32),
                "score": np.array([0.98], dtype=np.float32),
            },
            reward,
            terminated,
            truncated,
            info,
        )


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Policy head (action logits) and value head
        self.policy_head = torch.nn.Linear(10, self.action_space.n)
        self.value_head = torch.nn.Linear(10, 1)

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        print("obs in the implicity defined forward function is: ", obs)
        return super().forward(obs, deterministic)


env = CustomEnv()
env = FlattenObservation(env)
check_env(env)

model = PPO(
    CustomPolicy,
    env,
    verbose=1,
    n_steps=8,
    batch_size=8,
    n_epochs=1,
).learn(10)