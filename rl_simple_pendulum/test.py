import numpy as np
import gymnasium as gym

from pendulum_env import (
    Simulator,
    PendulumPlant,
    SimplePendulumEnv
)

from gymnasium.wrappers import FlattenObservation

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.sac.policies import MlpPolicy

from stable_baselines3 import PPO, SAC


# Simulator, Gym Env Params
class Parameters():
    def __init__(self):
        # dynamics parameters
        self.torque_limit = 1.5
        self.mass = 0.57288
        self.length = 0.5
        self.damping = 0.10
        self.gravity = 9.81
        self.coulomb_fric = 0.0
        self.inertia = self.mass*self.length**2
        self.use_symmetry = False

        # environment parameters
        self.dt = 0.01
        self.integrator = "runge_kutta"
        self.max_steps = 1000
        # continuous, discrete, soft_binary, soft_binary_with_repellor, 
        # open_ai_gym, open_ai_gym_red_torque
        self.reward_type = "soft_binary_with_repellor" 
        self.target = [np.pi, 0]
        self.target_epsilon = [0.1, 0.1]
        self.random_init = "everywhere" # False, start_vicinity, everywhere
        self.scale_action = True
        self.state_representation = 2

        # model params
        self.learning_rate = 0.0003
        self.training_timesteps=1e6
        self.reward_threshold=1000.0
        self.eval_frequency=10000
        self.n_eval_episodes=20
        self.verbose = 1
        self.device = 'cuda'

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    env = gym.make('gymnasium_env/SimplePendulum-v0')
    check_env(env)
    num_cpu = 4

    params = Parameters()
    pendulum = PendulumPlant(
        mass=params.mass,
        length=params.length,
        damping=params.damping,
        gravity=params.gravity,
        coulomb_fric=params.coulomb_fric,
        inertia=params.inertia,
        torque_limit=params.torque_limit
    )
    sim = Simulator(plant=pendulum)

    vec_env = make_vec_env(
        SimplePendulumEnv,
        env_kwargs={
            "simulator": sim,
            "max_steps": params.max_steps,
            "reward_type": params.reward_type,
            "dt": params.dt,
            "integrator": params.integrator,
            # [position,velocity] / [cos(position),sin(position),velocity]
            "state_representation": params.state_representation,
            "scale_action": params.scale_action,
            "random_init": params.random_init
        },
        n_envs=4,
        seed=0,
        vec_env_cls=SubprocVecEnv,
    )

    # vec_env = SubprocVecEnv([
    #     make_env("SimplePendulum-v0", i) for i in range(num_cpu)
    # ])

    # vec_env = make_vec_env(
    #     'SimplePendulum-v0',
    #     n_envs=12,
    #     seed=0,
    #     vec_env_cls=SubprocVecEnv,
    # )

    model = SAC(
        # MlpPolicy,
        'MlpPolicy',
        env=vec_env,
        verbose=1,
        batch_size=8,
        device="cuda",
        # n_steps=8,
        # n_epochs=1,
    )

    model.learn(
        total_timesteps=10,
        reset_num_timesteps=False,
        progress_bar=True,
    )


    # print(f"{vec_env.observation_space.shape=}")
    # print(f"{vec_env.observation_space.sample()=}")
    # print(f"{vec_env.action_space.shape=}")
    # print(f"{vec_env.action_space.sample()=}")
    # print()

    # env2 = FlattenObservation(pendulum_env.SimplePendulumEnv())
    # print(f"{env2.observation_space.shape=}")
    # print(f"{env2.observation_space.sample()=}")
    # print(f"{env2.action_space.shape=}")
    # print(f"{env2.action_space.sample()=}")

if __name__ == '__main__':
    # ref: ConnectionResetError: [Errno 104] Connection reset by peer
    main()