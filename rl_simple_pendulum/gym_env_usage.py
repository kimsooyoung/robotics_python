import numpy as np
import gymnasium as gym

import pendulum_env
from pendulum_env import (
    Simulator,
    PendulumPlant,
    SimplePendulumEnv
)

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
learning_rate=0.0003

# # Style1. Manual Env creation
# env = SimplePendulumEnv(
#     simulator=sim,
#     max_steps=max_steps,
#     reward_type=reward_type,
#     dt=dt,
#     integrator=integrator,
#     state_representation=2, # [position,velocity] / [cos(position),sin(position),velocity]
#     scale_action=True,
#     random_init=random_init,
#     render_mode="human" # human/rgb_array
# )

# Style2. gym api
env = gym.make(
    'gymnasium_env/SimplePendulum-v0',
    simulator=sim,
    max_steps=max_steps,
    reward_type=reward_type,
    dt=dt,
    integrator=integrator,
    state_representation=2, # [position,velocity] / [cos(position),sin(position),velocity]
    scale_action=True,
    random_init=random_init,
    render_mode="human" # human/rgb_array
)

# options={"state": np.array([1.0, 0.0]), "random_init": False}
options={"random_init": "everywhere"} # False/start_vicinity/everywhere

observation, info = env.reset(seed=None, options=options)
episode_over = False

while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"{observation=}")
    env.render()

    if terminated or truncated:
        episode_over = True

env.close()