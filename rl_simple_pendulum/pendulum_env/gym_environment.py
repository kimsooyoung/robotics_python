"""
original code from: https://github.com/dfki-ric-underactuated-lab/torque_limited_simple_pendulum
Gym Environment
===============
"""


# Other imports
import numpy as np
import gymnasium as gym
from typing import Optional

class SimplePendulumEnv(gym.Env):
    """
    An environment for reinforcement learning
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self,
                 simulator,
                 render_mode: Optional[str] = None,
                 max_steps=5000,
                 target=[np.pi, 0.0],
                 state_target_epsilon=[1e-2, 1e-2],
                 reward_type='continuous',
                 dt=1e-3,
                 integrator='runge_kutta',
                 state_representation=2,
                 validation_limit=-150,
                 scale_action=True,
                 random_init="False"):
        """
        An environment for reinforcement learning.

        Parameters
        ----------
        simulator : simulator object
        max_steps : int, default=5000
            maximum steps the agent can take before the episode
            is terminated
        target : array-like, default=[np.pi, 0.0]
            the target state of the pendulum
        state_target_epsilon: array-like, default=[1e-2, 1e-2]
            target epsilon for discrete reward type
        reward_type : string, default='continuous'
            the reward type selects the reward function which is used
            options are: 'continuous', 'discrete', 'soft_binary',
                         'soft_binary_with_repellor', 'open_ai_gym'
        dt : float, default=1e-3
            timestep for the simulation
        integrator : string, default='runge_kutta'
            the integrator which is used by the simulator
            options : 'euler', 'runge_kutta'
        state_representation : int, default=2
            determines how the state space of the pendulum is represented
            2 means state = [position, velocity]
            3 means state = [cos(position), sin(position), velocity]
        validation_limit : float, default=-150
            If the reward during validation episodes surpasses this value
            the training stops early
        scale_action : bool, default=True
            whether to scale the output of the model with the torque limit
            of the simulator's plant.
            If True the model is expected so return values in the intervall
            [-1, 1] as action.
        random_init : string, default="False"
            A string determining the random state initialisation
            "False" : The pendulum is set to [0, 0],
            "start_vicinity" : The pendulum position and velocity
                               are set in the range [-0.31, -0.31],
            "everywhere" : The pendulum is set to a random state in the whole
                           possible state space
        """
        self.simulator = simulator
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.target = target
        self.target[0] = self.target[0] % (2*np.pi)
        self.state_target_epsilon = state_target_epsilon
        self.reward_type = reward_type
        self.dt = dt
        self.integrator = integrator
        self.state_representation = state_representation
        self.validation_limit = validation_limit
        self.scale_action = scale_action
        self.random_init = random_init

        self.torque_limit = simulator.plant.torque_limit

        if state_representation == 2:
            # state is [th, vel]
            self.low = np.array([-6*2*np.pi, -8])
            self.high = np.array([6*2*np.pi, 8])
            self.observation_space = gym.spaces.Box(
                self.low, self.high, dtype=np.float32
            )
        elif state_representation == 3:
            # state is [cos(th), sin(th), vel]
            self.low = np.array([-1., -1., -8.])
            self.high = np.array([1., 1., 8.])
            self.observation_space = gym.spaces.Box(
                self.low, self.high, dtype=np.float32
            )

        if scale_action:
            self.action_space = gym.spaces.Box(-1, 1, shape=(1,))
        else:
            self.action_space = gym.spaces.Box(
                -self.torque_limit, self.torque_limit, shape=(1,)
            )

        self.state_shape = self.observation_space.shape
        self.n_actions = self.action_space.shape[0]
        self.n_states = self.observation_space.shape[0]
        self.action_limits = [-self.torque_limit, self.torque_limit]

        self.simulator.set_state(0, [0.0, 0.0])
        self.step_count = 0

    def step(self, action):
        """
        Take a step in the environment.

        Parameters
        ----------
        action : float
            the torque that is applied to the pendulum

        Returns
        -------
        observation : array-like
            the observation from the environment after the step
        reward : float
            the reward received on this step
        done : bool
            whether the episode has terminated
        info : dictionary
            may contain additional information
            (empty at the moment)
        """

        if self.scale_action:
            a = float(self.torque_limit * action)  # rescaling the action
        else:
            a = float(action)

        self.simulator.step(a, self.dt, self.integrator)

        # current_state is [position, velocity]
        # current_t, current_state = self.simulator.get_state()
        reward = self._calc_swingup_reward(a)
        
        observation = self._get_obs()
        done = self.check_final_condition()
        truncated = False
        info = self._get_info()

        self.step_count += 1

        return observation, reward, done, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment. The pendulum is initialized with a random state
        in the vicinity of the stable fixpoint
        (position and velocity are in the range[-0.31, 0.31])

        Parameters
        ----------
        options["state"] : array-like, default=None
            the state to which the environment is reset
            if state==None it defaults to the random initialisation
        options["random_init"] : string, default=None
            A string determining the random state initialisation
            if None, defaults to self.random_init
            "False" : The pendulum is set to [0, 0],
            "start_vicinity" : The pendulum position and velocity
                               are set in the range [-0.31, -0.31],
            "everywhere" : The pendulum is set to a random state in the whole
                           possible state space

        Returns
        -------
        observation : array-like
            the state the pendulum has been initilized to

        Raises:
        -------
        NotImplementedError
            when state==None and random_init does not indicate
            one of the implemented initializations
        """
        super().reset(seed=seed)

        if options is None:
            state = None
            random_init = "False" # TODO: everywhere?
        else:
            state = options.get("state") if "state" in options else None
            random_init = options.get("random_init") if "random_init" in options else None

        self.simulator.reset_data_recorder()
        self.step_count = 0
        if state is not None:
            init_state = np.copy(state)
        else:
            if random_init is None:
                random_init = self.random_init
            if random_init == "False":
                init_state = np.array([0.0, 0.0])
            elif random_init == "start_vicinity":
                pos_range = np.pi/10
                vel_range = np.pi/10
                init_state = np.array([\
                    np.random.rand()*2*pos_range - pos_range,
                    np.random.rand()*2*vel_range - vel_range
                ])
            elif random_init == "everywhere":
                pos_range = np.pi
                vel_range = 1.0
                init_state = np.array([
                    np.random.rand()*2*pos_range - pos_range,
                    np.random.rand()*2*vel_range - vel_range]
                )
            else:
                raise NotImplementedError(
                    f'Sorry, random initialization {random_init} ' +
                    'is not implemented.')

        self.simulator.set_state(0, init_state)
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self, mode='human'):
        # TODO: 
        pass

    def close(self):
        pass

    def _get_info(self):
        return None

    # some helper methods
    def _get_obs(self):
        """
        Transform the state from the simulator an observation by
        wrapping the position to the observation space.
        If state_representation==3 also transforms the state to the
        trigonometric value form.

        Parameters
        ----------
        state : array-like
            state as output by the simulator

        Returns
        -------
        observation : array-like
            observation in environment format
        """
        _, current_state = self.simulator.get_state()
        st = np.copy(current_state)
        st[1] = np.clip(st[1], self.low[-1], self.high[-1])

        if self.state_representation == 2:
            observation = np.array(
                [obs for obs in st], dtype=np.float32
            )
            # wraps the angle [−6π,6π)
            observation[0] = (observation[0] + 6*np.pi) % (np.pi*6*2) - 6*np.pi
        elif self.state_representation == 3:
            observation = np.array(
                [np.cos(st[0]), np.sin(st[0]), st[1]], dtype=np.float32
            )

        return observation

    def get_state_from_observation(self, obs):
        """
        Transform the observation to a pendulum state.
        Does nothing for state_representation==2.
        If state_representation==3 transforms trigonometric form
        back to regular form.

        Parameters
        ----------
        obs : array-like
            observation as received from get_observation

        Returns
        -------
        state : array-like
            state in simulator form
        """
        if self.state_representation == 2:
            state = np.copy(obs)
        elif self.state_representation == 3:
            state = np.array([np.arctan2(obs[1], obs[0]), obs[2]])
        return state

    def _calc_swingup_reward(self, action):
        """
        Calculate the reward for the pendulum for swinging up to the instable
        fixpoint. The reward function is selected based on the reward type
        defined during the object inizialization.

        Parameters
        ----------
        action : array-like
            action from controller

        Returns
        -------
        reward : float
            the reward for swinging up

        Raises
        ------
        NotImplementedError
            when the requested reward_type is not implemented

        """
        
        reward = None
        _, observation = self.simulator.get_state()
        
        pos = observation[0] % (2*np.pi)
        pos_diff = self.target[0] - pos
        pos_diff = np.abs((pos_diff + np.pi) % (np.pi * 2) - np.pi)
        
        vel = np.clip(observation[1], self.low[-1], self.high[-1])

        if self.reward_type == 'continuous':
            reward = - np.linalg.norm(pos_diff)
        elif self.reward_type == 'discrete':
            reward = np.float(
                np.linalg.norm(pos_diff) < self.state_target_epsilon[0]
            )
        elif self.reward_type == 'soft_binary':
            reward = np.exp(-pos_diff**2/(2*0.25**2))
        elif self.reward_type == 'soft_binary_with_repellor':
            reward = np.exp(-pos_diff ** 2 / (2 * 0.25 ** 2))
            pos_diff_repellor = pos - 0
            reward -= np.exp(-pos_diff_repellor ** 2 / (2 * 0.25 ** 2))
        elif self.reward_type == "open_ai_gym":
            # ref: https://github.com/Farama-Foundation/Gymnasium/blob/f02a56cf84a8bb3ebcdae392818c750ba1a2e4dc/gymnasium/envs/classic_control/pendulum.py#L138
            vel_diff = self.target[1] - vel
            reward = (
                -(pos_diff)**2.0 - 0.1*(vel_diff)**2.0 - 0.001*action**2.0
            )
        elif self.reward_type == "open_ai_gym_red_torque":
            vel_diff = self.target[1] - vel
            reward = (
                -(pos_diff)**2.0 - 0.1*(vel_diff)**2.0 - 0.01*action**2.0
            )
        else:
            raise NotImplementedError(
                f'Sorry, reward type {self.reward_type} is not implemented.')

        return reward

    def check_final_condition(self):
        """
        Checks whether a terminating condition has been met.
        The only terminating condition for the pendulum is if the maximum
        number of steps has been reached.

        Returns
        -------
        done : bool
            whether a terminating condition has been met
        """
        done = False
        if self.step_count > self.max_steps:
            done = True

        return done

    def is_goal(self, obs):
        """
        Checks whether an observation is in the goal region.
        The goal region is specified by the target and state_target_epsilon
        parameters in the class initialization.

        Parameters
        ----------
        obs : array-like
            observation as received from get_observation

        Returns
        -------
        goal : bool
            whether to observation is in the goal region
        """
        goal = False
        state = self.get_state_from_observation(obs)

        pos = state[0] % (2*np.pi)
        vel = np.clip(state[1], self.low[-1], self.high[-1])

        pos_diff = self.target[0] - pos
        pos_diff = np.abs((pos_diff + np.pi) % (np.pi * 2) - np.pi)
        vel_diff = self.target[1] - vel

        if np.abs(pos_diff) < self.state_target_epsilon[0] and \
           np.abs(vel_diff) < self.state_target_epsilon[1]:
            goal = True
        return goal

    def validation_criterion(self, validation_rewards,
                             final_obs=None, criterion=None):
        """
        Checks whether a list of rewards and optionally final observations
        fulfill the validation criterion.
        The validation criterion is fulfilled if the mean of the
        validation_rewards id greater than criterion.
        If final obs is also given, at least 90% of the observations
        have to be in the goal region.

        Parameters
        ----------
        validation_rewards : array-like
            A list of rewards (floats).
        final_obs : array-like, default=None
            A list of final observations.
            If None final observations are not considered.
        criterion: float, default=None
            The reward limit which has to be surpassed.

        Returns
        -------
        passed : bool
            Whether the rewards pass the validation test
        """
        if criterion is None:
            criterion = self.validation_limit

        N = len(validation_rewards)

        goal_reached = 0
        if final_obs is not None:
            for f in final_obs:
                if self.is_goal(f):
                    goal_reached += 1
        else:
            goal_reached = N

        passed = False
        if np.mean(validation_rewards) > criterion:
            if goal_reached/N > 0.9:
                passed = True

        n_passed = np.count_nonzero(np.asarray(validation_rewards) > criterion)

        print("Validation: ", end="")
        print(n_passed, "/", str(N), " passed reward limit, ", end="")
        print("Mean reward: ", np.mean(validation_rewards), ", ", end="")
        print(goal_reached, "/", len(final_obs), " found target state")
        return passed
