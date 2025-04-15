import os
import time
import argparse
import numpy as np
import gymnasium as gym
import stable_baselines3

from tqdm import tqdm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv
)
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

from pendulum_env import (
    Simulator,
    PendulumPlant,
    SimplePendulumEnv
)

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
        self.reward_type = "open_ai_gym" 
        self.target = [np.pi, 0]
        self.target_epsilon = [0.1, 0.1]
        self.random_init = "everywhere" # False, start_vicinity, everywhere
        self.scale_action = True
        self.state_representation = 2

        # model params
        self.learning_rate = 0.0003
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


def train(args, algo, sim, params):
    
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
        n_envs=args.num_parallel_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )

    eval_env = gym.make(
        'gymnasium_env/SimplePendulum-v0',
        simulator=sim,
        max_steps=params.max_steps,
        reward_type=params.reward_type,
        dt=params.dt,
        integrator=params.integrator,
        # [position,velocity] / [cos(position),sin(position),velocity]
        state_representation=params.state_representation,
        scale_action=params.scale_action,
        random_init="False"
    )

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"{MODEL_DIR}/{args.algorithm}/{train_time}"
    log_path = f"{LOG_DIR}"

    print(
        f"Training on {args.num_parallel_envs} parallel training environments and saving models to '{model_path}'"
    )

    if args.model_path is not None:
        model = algo.load(
            path=args.model_path, 
            env=vec_env,
            device=params.device,
            verbose=params.verbose,
            tensorboard_log=log_path
        )
    else:
        model = algo(
            'MlpPolicy', # Must be str
            vec_env,
            device=params.device,
            verbose=params.verbose,
            tensorboard_log=log_path,
            learning_rate=params.learning_rate
        )

    # define training callbacks
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=params.reward_threshold,
        verbose=params.verbose
    )

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
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=f"{args.algorithm}/{train_time}",
        callback=eval_callback
    )

    # Save final model
    model.save(f"{model_path}/final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        default='SAC',
        help="Custom name of the run. Note that all runs are saved in the 'models' directory and have the training time prefixed.",
    )
    parser.add_argument(
        "--num_parallel_envs",
        type=int,
        default=3,
        help="Number of parallel environments while training",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=3_000_000,
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

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    train(args, sb3_class, sim, params)

# Usage
# python3 main_training.py --algorithm PPO --total_timesteps 10
# python3 main_training.py --algorithm SAC 
# python3 main_training.py --algorithm A2C 
# python3 main_training.py --algorithm TD3 

# tensorboard --logdir <log-file>

# pip install moviepy
# pip install tensorboard
# gymnasium 0.29.1
# stable_baselines3 2.6.0