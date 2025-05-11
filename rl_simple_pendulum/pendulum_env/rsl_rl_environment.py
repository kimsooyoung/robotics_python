# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch
from typing import Optional

from rsl_rl.env import VecEnv


class RslRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for RSL-RL library

    To use asymmetric actor-critic, the environment instance must have the attributes :attr:`num_privileged_obs` (int).
    This is used by the learning agent to allocate buffers in the trajectory memory. Additionally, the returned
    observations should have the key "critic" which corresponds to the privileged observations. Since this is
    optional for some environments, the wrapper checks if these attributes exist. If they don't then the wrapper
    defaults to zero as number of privileged observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: gym.Env, clip_actions: Optional[float] = None):
        """Initializes the wrapper.


        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """

        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        self.device = torch.device("cuda")
        self.num_envs = 1
        self.num_actions = 1
        self.max_episode_length = self.unwrapped.max_steps
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device)

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def render_mode(self):
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self):
        """Returns the current observations of the environment."""
        observation = self.unwrapped._get_obs().reshape(1, -1)
        observation = torch.from_numpy(observation).float()  # Convert to torch.Tensor
        obs_dict = {"policy": observation}
        return obs_dict["policy"], {"observations": obs_dict}

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self):
        observation, info = self.env.reset()
        return observation, info
    
    def step(self, actions):
        # # clip actions
        # if self.clip_actions is not None:
        #     actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        
        # record step information
        observation, reward, terminated, truncated, extras = self.env.step(actions)
        
        # # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated)
        # .to(dtype=torch.long)
        # # move extra observations to the extras dict
        # obs = obs_dict["policy"]
        # extras["observations"] = obs_dict
        # # move time out information to the extras dict
        # # this is only needed for infinite horizon tasks
        # if not self.unwrapped.cfg.is_finite_horizon:
        #     extras["time_outs"] = truncated

        observation = torch.tensor(observation, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        obs_dict = {"policy": observation}
        extras["observations"] = obs_dict

        # return the step information
        return observation, reward, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()
