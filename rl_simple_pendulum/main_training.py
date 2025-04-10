import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

from pendulum_env import Simulator
from pendulum_env import PendulumPlant
from pendulum_env import SimplePendulumEnv


# get the simulator
torque_limit = 5.0
mass = 0.57288
length = 0.5
damping = 0.10
gravity = 9.81
coulomb_fric = 0.0
inertia = mass*length**2

pendulum = PendulumPlant(
    mass=mass,
    length=length,
    damping=damping,
    gravity=gravity,
    coulomb_fric=coulomb_fric,
    inertia=inertia,
    torque_limit=torque_limit
)

sim = Simulator(plant=pendulum)

# environment parameters
dt = 0.01
integrator = "runge_kutta"
max_steps = 1000
reward_type = "soft_binary_with_repellor"
# reward_type = "open_ai_gym"
target = [np.pi, 0]
target_epsilon = [0.1, 0.1]
random_init = "False"

env = SimplePendulumEnv(
    simulator=sim,
    max_steps=max_steps,
    reward_type=reward_type,
    dt=dt,
    integrator=integrator,
    state_representation=2, # [position,velocity] / [cos(position),sin(position),velocity]
    scale_action=True,
    random_init=random_init
)

log_dir = "./logs"
tensorboard_log = os.path.join(log_dir, "tb_logs")
agent = SAC(
    MlpPolicy,
    env,
    verbose=1,
    tensorboard_log=tensorboard_log,
    learning_rate=learning_rate
)

# options={"state": np.array([0.0, 0.0]), "random_init": False}
options={"random_init": "start_vicinity"} # False/start_vicinity/everywhere

observation = env.reset()
episode_over = False

while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"{observation=}")

    if terminated:
        episode_over = True

env.close()