import os
import time
import argparse
import numpy as np
import gymnasium as gym
import stable_baselines3

from tqdm import tqdm
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

from pendulum_env import Simulator
from pendulum_env import PendulumPlant
from pendulum_env import SimplePendulumEnv

MODEL_DIR = "models"
LOG_DIR = "logs"

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


def get_model_class(algo_name):
    """Retrieve the SB3 algorithm class dynamically."""
    try:
        print(f"You are using {algo_name} from sb3")
        return getattr(stable_baselines3, algo_name)
    except AttributeError:
        raise ValueError(f"Invalid algorithm: {algo_name}. Available options: A2C, DDPG, PPO, SAC, TD3")

def test(args, algo, sim, params):

    env = SimplePendulumEnv(
        simulator=sim,
        max_steps=params.max_steps,
        reward_type=params.reward_type,
        dt=params.dt,
        integrator=params.integrator,
        state_representation=params.state_representation, # [position,velocity] / [cos(position),sin(position),velocity]
        scale_action=params.scale_action,
        random_init=params.random_init,
        render_mode=args.render_mode # human/rgb_array
    )
    if args.render_mode == "rgb_array":
        env = gym.wrappers.RecordVideo(
            env=env, 
            video_folder="video", name_prefix="test-video", 
            episode_trigger=lambda x: x % 2 == 0
        )

    model = algo.load(
        path=args.model_path, 
        env=env, 
        device=params.device,
        verbose=params.verbose
    )

    num_episodes = args.num_test_episodes
    total_reward = 0
    total_length = 0

    for _ in tqdm(range(num_episodes)):

        observation, _ = env.reset()

        ep_len = 0
        ep_reward = 0
        while True:
            theta, omega = observation

            action, states = model.predict(observation)
            print(f"{action=} {observation=}")
            
            observation, reward, terminated, truncated, info = env.step(action)

            env.render()
            
            ep_reward += reward
            ep_len += 1

            if terminated or truncated:
                print(f"{ep_len=}  {ep_reward=}")
                break

        total_length += ep_len
        total_reward += ep_reward

    print(
        f"Avg episode reward: {total_reward / num_episodes}, avg episode length: {total_length / num_episodes}"
    )

    if args.render_mode == "rgb_array":
        env.close_video_recorder()

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        default='SAC',
        help="Custom name of the run. Note that all runs are saved in the 'models' directory and have the training time prefixed.",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        help="Render Simulation (human) or Record video (rgb_array).",
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=1,
        help="Number of episodes to test the model",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model (.zip). If passed for training, the model is used as the starting point for training. If passed for testing, the model is used for inference.",
    )
    args = parser.parse_args()

    try:
        sb3_class = get_model_class(args.algorithm)
    except ValueError as e:
        print(e)
        exit(1)

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

    if args.model_path is None:
        raise ValueError("--model_path is required for testing")
    
    test(args, sb3_class, sim, params)

# TODO: PPO
# python3 main_evaluation.py --algorithm PPO --model_path <sth> --render_mode <human>
# python3 main_evaluation.py --algorithm A2C --model_path 

# pip install moviepy
# pip install tensorboard
# gymnasium 0.29.1
# stable_baselines3 2.6.0