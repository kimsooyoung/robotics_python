import os
import time
import argparse
import numpy as np
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner


from pendulum_env import (
    Simulator,
    PendulumPlant,
    SimplePendulumEnv,
    RslRlVecEnvWrapper
)

MODEL_DIR = "rsl_rl_models"
LOG_DIR = "rsl_rl_logs"

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


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def test():

    env = gym.make(
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
    print(type(env), type(env.unwrapped))
    rsl_env = RslRlVecEnvWrapper(env, clip_actions=0.5)

def train(sim, params):
    
    env = gym.make(
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
    rsl_env = RslRlVecEnvWrapper(env, clip_actions=0.5)


    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    train_cfg = get_train_cfg(train_time, params.max_steps)
    log_dir = f"{LOG_DIR}"

    runner = OnPolicyRunner(
        rsl_env, 
        train_cfg, 
        log_dir, 
        device="cuda:0"
    )

    runner.learn(
        num_learning_iterations=params.max_steps, 
        init_at_random_ep_len=True
    )


if __name__ == "__main__":

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

    # test()
    train(sim, params)

# Usage
# python3 main_training_rsl_rl.py --algorithm PPO --total_timesteps 10
# python3 main_training.py --algorithm SAC 
# python3 main_training.py --algorithm A2C 
# python3 main_training.py --algorithm TD3 

# tensorboard --logdir <log-file>

# pip install moviepy
# pip install tensorboard
# gymnasium 0.29.1
# stable_baselines3 2.6.0