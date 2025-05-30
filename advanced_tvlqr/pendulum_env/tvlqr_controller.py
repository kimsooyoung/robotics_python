"""
tvLQR Controller
================
"""


# Global imports
import numpy as np
from pydrake.all import (
    FiniteHorizonLinearQuadraticRegulatorOptions,
    FiniteHorizonLinearQuadraticRegulator,
    PiecewisePolynomial,
    Linearize,
    LinearQuadraticRegulator,
)

# from pydrake.examples.pendulum import PendulumPlant
from pydrake.examples import PendulumPlant


class TVLQRController:
    """
    Controller acts on a predefined trajectory.
    """

    def __init__(
        self,
        data_dict,
        mass=1.0,
        length=0.5,
        damping=0.1,
        gravity=9.81,
        torque_limit=np.inf,
    ):
        """
        Controller acts on a predefined trajectory.

        Parameters
        ----------
        data_dict : dictionary
            a dictionary containing the trajectory to follow
            should have the entries:
            data_dict["des_time"] : desired timesteps
            data_dict["des_pos"] : desired positions
            data_dict["des_vel"] : desired velocities
            data_dict["des_tau"] : desired torques
        mass : float, default=1.0
            mass of the pendulum [kg]
        length : float, default=0.5
            length of the pendulum [m]
        damping : float, default=0.1
            damping factor of the pendulum [kg m/s]
        gravity : float, default=9.81
            gravity (positive direction points down) [m/s^2]
        torque_limit : float, default=np.inf
            the torque_limit of the pendulum actuator
        """

        # load the trajectory
        self.traj_time = data_dict["des_time"]
        self.traj_pos = data_dict["des_pos"]
        self.traj_vel = data_dict["des_vel"]
        self.traj_tau = data_dict["des_tau"]

        self.max_time = self.traj_time[-1]

        self.traj_time = np.reshape(self.traj_time, (self.traj_time.shape[0], -1))
        self.traj_tau = np.reshape(self.traj_tau, (self.traj_tau.shape[0], -1)).T

        x0_desc = np.vstack((self.traj_pos, self.traj_vel))
        # u0_desc = self.traj_tau

        u0 = PiecewisePolynomial.FirstOrderHold(self.traj_time, self.traj_tau)
        x0 = PiecewisePolynomial.CubicShapePreserving(
            self.traj_time, x0_desc, zero_end_point_derivatives=True
        )

        # create drake pendulum plant
        self.plant = PendulumPlant()
        self.context = self.plant.CreateDefaultContext()
        params = self.plant.get_mutable_parameters(self.context)
        params[0] = mass
        params[1] = length
        params[2] = damping
        params[3] = gravity

        self.torque_limit = torque_limit

        # create lqr context
        self.tilqr_context = self.plant.CreateDefaultContext()
        self.plant.get_input_port(0).FixValue(self.tilqr_context, [0])
        # self.Q_tilqr = np.diag((50., 1.))
        self.Q_tilqr = np.diag((10.0, 1.0))
        self.R_tilqr = [1]

        # Setup Options and Create TVLQR
        self.options = FiniteHorizonLinearQuadraticRegulatorOptions()
        # self.Q = np.diag([200., 0.1])
        self.Q = np.diag([10.0, 1.0])
        self.options.x0 = x0
        self.options.u0 = u0

        self.counter = 0
        self.last_pos = 0.0
        self.last_vel = 0.0

    def init(self, x0):
        self.counter = 0
        self.last_pos = 0.0
        self.last_vel = 0.0

    def set_goal(self, x):
        self.goal = x
        pos = x[0] + np.pi
        pos = (pos + np.pi) % (2 * np.pi) - np.pi
        self.tilqr_context.SetContinuousState([pos, x[1]])
        linearized_pendulum = Linearize(self.plant, self.tilqr_context)
        (self.K, S) = LinearQuadraticRegulator(
            linearized_pendulum.A(), linearized_pendulum.B(), self.Q_tilqr, self.R_tilqr
        )

        self.options.Qf = S

        self.tvlqr = FiniteHorizonLinearQuadraticRegulator(
            self.plant,
            self.context,
            t0=self.options.u0.start_time(),
            tf=self.options.u0.end_time(),
            Q=self.Q,
            R=np.eye(1) * 2,
            options=self.options,
        )

    def get_control_output(
        self, meas_pos=None, meas_vel=None, meas_tau=None, meas_time=None
    ):
        """
        The function to read and send the entries of the loaded trajectory
        as control input to the simulator/real pendulum.

        Parameters
        ----------
        meas_pos : float, deault=None
            the position of the pendulum [rad]
        meas_vel : float, deault=None
            the velocity of the pendulum [rad/s]
        meas_tau : float, deault=None
            the meastured torque of the pendulum [Nm]
        meas_time : float, deault=None
            the collapsed time [s]

        Returns
        -------
        des_pos : float
            the desired position of the pendulum [rad]
        des_vel : float
            the desired velocity of the pendulum [rad/s]
        des_tau : float
            the torque supposed to be applied by the actuator [Nm]
        """

        pos = float(np.squeeze(meas_pos))
        vel = float(np.squeeze(meas_vel))
        x = np.array([[pos], [vel]])

        des_pos = self.last_pos
        des_vel = self.last_vel
        # des_tau = 0

        if self.counter < len(self.traj_pos):
            des_pos = self.traj_pos[self.counter]
            des_vel = self.traj_vel[self.counter]
            # des_tau = self.traj_tau[self.counter]
            self.last_pos = des_pos
            self.last_vel = des_vel

        self.counter += 1

        time = min(meas_time, self.max_time)
        if meas_time <= self.max_time:
            uu = self.tvlqr.u0.value(time)
            xx = self.tvlqr.x0.value(time)
            KK = self.tvlqr.K.value(time)
            kk = self.tvlqr.k0.value(time)

            xdiff = x - xx
            pos_diff = (xdiff[0] + np.pi) % (2 * np.pi) - np.pi
            xdiff[0] = pos_diff

            des_tau = (uu - KK.dot(xdiff) - kk)[0][0]
        else:
            delta_pos = pos - self.goal[0]
            delta_pos_wrapped = (delta_pos + np.pi) % (2 * np.pi) - np.pi

            delta_y = np.asarray([delta_pos_wrapped, vel - self.goal[1]])

            des_tau = np.asarray(-self.K.dot(delta_y))[0]

        des_tau = np.clip(des_tau, -self.torque_limit, self.torque_limit)

        return des_pos, des_vel, des_tau

    def set_Qf(self, Qf):
        """
        This function is useful only for RoA purposes. Used to set the
        final S-matrix of the tvlqr controller.

        Parameters
        ----------
        Qf : matrix
            the S-matrix from time-invariant RoA estimation around the
            up-right position.
        """
        self.Qf = Qf
