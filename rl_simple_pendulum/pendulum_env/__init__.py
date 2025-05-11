from .simulation import Simulator
from .pendulum_plant import PendulumPlant
from .gym_environment import SimplePendulumEnv
from .rsl_rl_environment import RslRlVecEnvWrapper

from gymnasium.envs.registration import register

register(
    id="gymnasium_env/SimplePendulum-v0",
    entry_point=SimplePendulumEnv,
    kwargs={
        "simulator": None
    }
)