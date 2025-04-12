import os
import time
import argparse
import numpy as np
import gymnasium as gym

from pathlib import Path
from tqdm import tqdm

from stable_baselines3 import PPO, SAC
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

# TODO: 

# Simulator, Gym Env Params
class Parameters():
    def __init__(self):
        # dynamics parameters
        self.torque_limit = 5.0
        self.mass = 0.57288
        self.length = 0.5
        self.damping = 0.10
        self.gravity = 9.81
        self.coulomb_fric = 0.0
        self.inertia = self.mass*self.length**2

        # environment parameters
        # TODO: env params into argument
        self.dt = 0.01
        self.integrator = "runge_kutta"
        self.max_steps = 1000
        # reward_type = "open_ai_gym"
        self.reward_type = "soft_binary_with_repellor"
        self.target = [np.pi, 0]
        self.target_epsilon = [0.1, 0.1]
        self.random_init = "False"
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
    env = SimplePendulumEnv(
        simulator=sim,
        max_steps=params.max_steps,
        reward_type=params.reward_type,
        dt=params.dt,
        integrator=params.integrator,
        state_representation=params.state_representation, # [position,velocity] / [cos(position),sin(position),velocity]
        scale_action=params.scale_action,
        random_init=params.random_init
    )

    eval_env = SimplePendulumEnv(
        simulator=sim,
        max_steps=params.max_steps,
        reward_type=params.reward_type,
        dt=params.dt,
        integrator=params.integrator,
        state_representation=params.state_representation, # [position,velocity] / [cos(position),sin(position),velocity]
        scale_action=params.scale_action,
        random_init="False"
    )

    # TODO : vec env after register
    # vec_env = make_vec_env(
    #     Go1MujocoEnv,
    #     env_kwargs={"ctrl_type": args.ctrl_type},
    #     n_envs=args.num_parallel_envs,
    #     seed=args.seed,
    #     vec_env_cls=SubprocVecEnv,
    # )

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name is None:
        run_name = f"{train_time}"
    else:
        run_name = f"{train_time}-{args.run_name}"

    model_path = f"{MODEL_DIR}/{run_name}"
    print(
        f"Training on {args.num_parallel_envs} parallel training environments and saving models to '{model_path}'"
    )

    if args.model_path is not None:
        model = SAC.load(
            path=args.model_path, 
            env=env, 
            verbose=params.verbose,
            tensorboard_log=LOG_DIR
        )
    else:
        model = SAC(
            MlpPolicy,
            env,
            verbose=params.verbose,
            tensorboard_log=LOG_DIR,
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
    # # Evaluate the model every eval_frequency for 5 episodes and save
    # # it if it's improved over the previous best model.
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

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
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


def test(args):
    pass
    # model_path = Path(args.model_path)

    # if not args.record_test_episodes:
    #     # Render the episodes live
    #     env = Go1MujocoEnv(
    #         ctrl_type=args.ctrl_type,
    #         render_mode="human",
    #     )
    #     inter_frame_sleep = 0.016
    # else:
    #     # Record the episodes
    #     env = Go1MujocoEnv(
    #         ctrl_type=args.ctrl_type,
    #         render_mode="rgb_array",
    #         camera_name="tracking",
    #         width=1920,
    #         height=1080,
    #     )
    #     env = gym.wrappers.RecordVideo(
    #         env, video_folder="recordings/", name_prefix=model_path.parent.name
    #     )
    #     inter_frame_sleep = 0.0

    # model = PPO.load(path=model_path, env=env, verbose=1)

    # num_episodes = args.num_test_episodes
    # total_reward = 0
    # total_length = 0
    # for _ in tqdm(range(num_episodes)):
    #     obs, _ = env.reset()
    #     env.render()

    #     ep_len = 0
    #     ep_reward = 0
    #     while True:
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         ep_reward += reward
    #         ep_len += 1

    #         # Slow down the rendering
    #         time.sleep(inter_frame_sleep)

    #         if terminated or truncated:
    #             print(f"{ep_len=}  {ep_reward=}")
    #             break

    #     total_length += ep_len
    #     total_reward += ep_reward

    # print(
    #     f"Avg episode reward: {total_reward / num_episodes}, avg episode length: {total_length / num_episodes}"
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, choices=["train", "test"])
    parser.add_argument(
        "--run_name",
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
        "--num_test_episodes",
        type=int,
        default=5,
        help="Number of episodes to test the model",
    )
    parser.add_argument(
        "--record_test_episodes",
        action="store_true",
        help="Whether to record the test episodes or not. If false, the episodes are rendered in the window.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=5_000_000,
        help="Number of timesteps to train the model for",
    )
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=10_000,
        help="The frequency of evaluating the models while training",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model (.zip). If passed for training, the model is used as the starting point for training. If passed for testing, the model is used for inference.",
    )
    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["torque", "position"],
        default="position",
        help="Whether the model should control the robot using torque or position control.",
    )
    parser.add_argument("--seed", type=int, default=0)
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

    if args.run == "train":
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        train(args, sim, params)
    elif args.run == "test":
        if args.model_path is None:
            raise ValueError("--model_path is required for testing")
        test(args, sim, params)

# python3 train.py --run train

# pip install tensorboard
# gymnasium 0.29.1
# stable_baselines3 2.6.0