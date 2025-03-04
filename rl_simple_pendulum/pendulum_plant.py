"""
Pendulum Plant
==============
"""


import numpy as np
import yaml


class PendulumPlant:
    def __init__(self, mass=1.0, length=0.5, damping=0.1, gravity=9.81,
                 coulomb_fric=0.0, inertia=None, torque_limit=np.inf):

        """
        The PendulumPlant class contains the kinematics and dynamics
        of the simple pendulum.

        The state of the pendulum in this class is described by
            state = [angle, angular velocity]
            (array like with len(state)=2)
            in units: rad and rad/s
        The zero state of the angle corresponds to the pendulum hanging down.
        The plant expects an actuation input (tau) either as float or
        array like in units Nm.
        (in which case the first entry is used (which should be a float))

        Parameters
        ----------
        mass : float, default=1.0
            pendulum mass, unit: kg
        length : float, default=0.5
            pendulum length, unit: m
        damping : float, default=0.1
            damping factor (proportional to velocity), unit: kg*m/s
        gravity : float, default=9.81
            gravity (positive direction points down), unit: m/s^2
        coulomb_fric : float, default=0.0
            friction term, (independent of magnitude of velocity), unit: Nm
        inertia : float, default=None
            inertia of the pendulum (defaults to point mass inertia)
            unit: kg*m^2
        torque_limit: float, default=np.inf
            maximum torque that the motor can apply, unit: Nm
        """

        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        self.coulomb_fric = coulomb_fric
        if inertia is None:
            self.inertia = mass*length*length
        else:
            self.inertia = inertia

        self.torque_limit = torque_limit

        self.dof = 1
        self.n_actuators = 1
        self.base = [0, 0]
        self.n_links = 1
        self.workspace_range = [[-1.2*self.l, 1.2*self.l],
                                [-1.2*self.l, 1.2*self.l]]

    def load_params_from_file(self, filepath):
        """
        Load the pendulum parameters from a yaml file.

        Parameters
        ----------
        filepath : string
            path to yaml file
        """

        with open(filepath, 'r') as yaml_file:
            params = yaml.safe_load(yaml_file)
        self.m = params["mass"]
        self.l = params["length"]
        self.b = params["damping"]
        self.g = params["gravity"]
        self.coulomb_fric = params["coulomb_fric"]
        self.inertia = params["inertia"]
        self.torque_limit = params["torque_limit"]
        self.dof = params["dof"]
        self.n_actuators = params["n_actuators"]
        self.base = params["base"]
        self.n_links = params["n_links"]
        self.workspace_range = [[-1.2*self.l, 1.2*self.l],
                                [-1.2*self.l, 1.2*self.l]]

    def forward_kinematics(self, pos):

        """
        Computes the forward kinematics.

        Parameters
        ----------
        pos : float, angle of the pendulum

        Returns
        -------
        list : A list containing one list (for one end-effector)
              The inner list contains the x and y coordinates
              for the end-effector of the pendulum
        """

        ee_pos_x = float(self.l * np.sin(pos))
        ee_pos_y = float(-self.l * np.cos(pos))
        return [[ee_pos_x, ee_pos_y]]

    def inverse_kinematics(self, ee_pos):

        """
        Comutes inverse kinematics

        Parameters
        ----------
        ee_pos : array like,
            len(state)=2
            contains the x and y position of the end_effector
            floats, units: m

        Returns
        -------
        pos : float
            angle of the pendulum, unit: rad
        """

        pos = np.arctan2(ee_pos[0]/self.l, ee_pos[1]/(-1.0*self.l))
        return pos

    def forward_dynamics(self, state, tau):

        """
        Computes forward dynamics

        Parameters
        ----------
        state : array like
            len(state)=2
            The state of the pendulum [angle, angular velocity]
            floats, units: rad, rad/s
        tau : float
            motor torque, unit: Nm

        Returns
        -------
            - float, angular acceleration, unit: rad/s^2
        """

        torque = np.clip(tau, -np.asarray(self.torque_limit),
                         np.asarray(self.torque_limit))

        accn = (torque - self.m * self.g * self.l * np.sin(state[0]) -
                self.b * state[1] -
                np.sign(state[1]) * self.coulomb_fric) / self.inertia
        return accn

    def inverse_dynamics(self, state, accn):

        """
        Computes inverse dynamics

        Parameters
        ----------
        state : array like
            len(state)=2
            The state of the pendulum [angle, angular velocity]
            floats, units: rad, rad/s
        accn : float
            angular acceleration, unit: rad/s^2

        Returns
        -------
        tau : float
            motor torque, unit: Nm
        """

        tau = accn * self.inertia + \
            self.m * self.g * self.l * np.sin(state[0]) + \
            self.b*state[1] + np.sign(state[1]) * self.coulomb_fric
        return tau

    def rhs(self, t, state, tau):

        """
        Computes the integrand of the equations of motion.

        Parameters
        ----------
        t : float
            time, not used (the dynamics of the pendulum are time independent)
        state : array like
            len(state)=2
            The state of the pendulum [angle, angular velocity]
            floats, units: rad, rad/s
        tau : float or array like
            motor torque, unit: Nm

        Returns
        -------
        res : array like
              the integrand, contains [angular velocity, angular acceleration]
        """

        if isinstance(tau, (list, tuple, np.ndarray)):
            torque = tau[0]
        else:
            torque = tau

        accn = self.forward_dynamics(state, torque)

        res = np.zeros(2*self.dof)
        res[0] = state[1]
        res[1] = accn
        return res

    def potential_energy(self, state):
        Epot = self.m*self.g*self.l*(1-np.cos(state[0]))
        return Epot

    def kinetic_energy(self, state):
        Ekin = 0.5*self.m*(self.l*state[1])**2.0
        return Ekin

    def total_energy(self, state):
        E = self.potential_energy(state) + self.kinetic_energy(state)
        return E
