import os
import time
import argparse
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from stable_baselines3 import PPO, SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv
)
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


def train(args, sim, params):
    # env = SimplePendulumEnv(
    #     simulator=sim,
    #     max_steps=params.max_steps,
    #     reward_type=params.reward_type,
    #     dt=params.dt,
    #     integrator=params.integrator,
    #     state_representation=params.state_representation, # [position,velocity] / [cos(position),sin(position),velocity]
    #     scale_action=params.scale_action,
    #     random_init=params.random_init
    # )

    # eval_env = SimplePendulumEnv(
    #     simulator=sim,
    #     max_steps=params.max_steps,
    #     reward_type=params.reward_type,
    #     dt=params.dt,
    #     integrator=params.integrator,
    #     state_representation=params.state_representation, # [position,velocity] / [cos(position),sin(position),velocity]
    #     scale_action=params.scale_action,
    #     random_init="False"
    # )

    env = DummyVecEnv([lambda: SimplePendulumEnv(
        simulator=sim,
        max_steps=params.max_steps,
        reward_type=params.reward_type,
        dt=params.dt,
        integrator=params.integrator,
        state_representation=params.state_representation,
        scale_action=params.scale_action,
        random_init=params.random_init
    )])

    eval_env = DummyVecEnv([lambda: SimplePendulumEnv(
        simulator=sim,
        max_steps=params.max_steps,
        reward_type=params.reward_type,
        dt=params.dt,
        integrator=params.integrator,
        state_representation=params.state_representation,
        scale_action=params.scale_action,
        random_init="False"
    )])

    # TODO : vec env after register
    # vec_env = make_vec_env(
    #     Go1MujocoEnv,
    #     env_kwargs={"ctrl_type": args.ctrl_type},
    #     n_envs=args.num_parallel_envs,
    #     seed=args.seed,
    #     vec_env_cls=SubprocVecEnv,
    # )

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    # TODO: algo
    run_name = f"{train_time}"

    model_path = f"{MODEL_DIR}/{run_name}"
    log_path = f"{LOG_DIR}"

    print(
        f"Training on {args.num_parallel_envs} parallel training environments and saving models to '{model_path}'"
    )

    if args.model_path is not None:
        model = PPO.load(
            path=args.model_path, 
            env=env, 
            verbose=params.verbose,
            tensorboard_log=log_path
        )
    else:
        model = PPO(
            MlpPolicy,
            env,
            verbose=params.verbose,
            tensorboard_log=log_path,
            learning_rate=params.learning_rate
        )
        # TODO: PPO
        # model = PPO.load(
        #     path=args.model_path, 
        #     env=vec_env, 
        #     verbose=1, 
        #     tensorboard_log=LOG_DIR
        # )
    # else:
        # Default PPO model hyper-parameters give good results
        # TODO: Use dynamic learning rate
        # TODO: PPO
        # model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)

    # TODO: Vector Env

    # eval_callback = EvalCallback(
    #     vec_env,
    #     best_model_save_path=model_path,
    #     log_path=LOG_DIR,
    #     eval_freq=args.eval_frequency,
    #     n_eval_episodes=5,
    #     deterministic=True,
    #     render=False,
    # )

    # define training callbacks
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=params.reward_threshold,
        verbose=params.verbose
    )

    # log_path = os.path.join(log_dir, 'best_model')

    # Evaluate the model every eval_frequency for 'n_eval_episodes' episodes 
    # and save it if it's improved over the previous best model.
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=params.eval_frequency,
        n_eval_episodes=params.n_eval_episodes,
    )

    # train
    model.learn(
        total_timesteps=params.training_timesteps,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=run_name, # TODO: Check
        callback=eval_callback
    )

    # Save final model
    model.save(f"{model_path}/final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Custom name of the run. Note that all runs are saved in the 'models' directory and have the training time prefixed.",
    )
    parser.add_argument(
        "--num_parallel_envs",
        type=int,
        default=12,
        help="Number of parallel environments while training",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="Number of timesteps to train the model for",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model (.zip). If passed for training, the model is used as the starting point for training. If passed for testing, the model is used for inference.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0
    )
    args = parser.parse_args()

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

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    train(args, sim, params)

# TODO
# env register
# vector env
# 

# python3 main_training.py

# pip install moviepy
# pip install tensorboard
# gymnasium 0.29.1
# stable_baselines3 2.6.0