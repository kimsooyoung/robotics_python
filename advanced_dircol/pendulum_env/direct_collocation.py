# referenced from : https://github.com/dfki-ric-underactuated-lab/torque_limited_simple_pendulum
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import Solve, DirectCollocation
from pydrake.trajectories import PiecewisePolynomial
from pydrake.examples import PendulumPlant, PendulumState


class DirectCollocationCalculator:
    """
    Class to calculate a control trajectory with direct collocation.
    """

    def __init__(self):
        """
        Class to calculate a control trajectory with direct collocation.
        """
        self.pendulum_plant = PendulumPlant()
        self.pendulum_context = self.pendulum_plant.CreateDefaultContext()

    def init_pendulum(
        self, mass=0.57288, length=0.5, damping=0.15, gravity=9.81, torque_limit=2.0
    ):
        """
        Initialize the pendulum parameters.

        Parameters
        ----------
        mass : float, default=0.57288
            mass of the pendulum [kg]
        length : float, default=0.5
            length of the pendulum [m]
        damping : float, default=0.15
            damping factor of the pendulum [kg m/s]
        gravity : float, default=9.81
            gravity (positive direction points down) [m/s^2]
        torque_limit : float, default=2.0
            the torque_limit of the pendulum actuator
        """
        self.pendulum_params = self.pendulum_plant.get_mutable_parameters(
            self.pendulum_context
        )
        self.pendulum_params[0] = mass
        self.pendulum_params[1] = length
        self.pendulum_params[2] = damping
        self.pendulum_params[3] = gravity
        self.torque_limit = torque_limit

    def compute_trajectory(
        self,
        N=21,
        max_dt=0.5,
        start_state=[0.0, 0.0],
        goal_state=[np.pi, 0.0],
        initial_x_trajectory=None,
        control_cost=1.0,
    ):
        """
        Compute a trajectory from a start state to a goal state
        for the pendulum.

        Parameters
        ----------
        N : int, default=21
            number of collocation points
        max_dt : float, default=0.5
            maximum allowed timestep between collocation points
        start_state : array_like, default=[0.0, 0.0]
            the start state of the pendulum [position, velocity]
        goal_state : array_like, default=[np.pi, 0.0]
            the goal state for the trajectory
        initial_x_trajectory : array-like, default=None
            initial guess for the state space trajectory
            ignored if None

        Returns
        -------
        x_trajectory : pydrake.trajectories.PiecewisePolynomial
            trajectory in state space
        dircol : pydrake.systems.trajectory_optimization.DirectCollocation
            DirectCollocation pydrake object
        result : pydrake.solvers.mathematicalprogram.MathematicalProgramResult
            MathematicalProgramResult pydrake object
        """
        dircol = DirectCollocation(
            self.pendulum_plant,
            self.pendulum_context,
            num_time_samples=N,
            minimum_time_step=0.05,
            maximum_time_step=max_dt,
            input_port_index=self.pendulum_plant.get_input_port().get_index(),
        )

        dircol.AddEqualTimeIntervalsConstraints()

        u = dircol.input()
        dircol.AddConstraintToAllKnotPoints(-self.torque_limit <= u[0])
        dircol.AddConstraintToAllKnotPoints(u[0] <= self.torque_limit)

        initial_state = PendulumState()
        initial_state.set_theta(start_state[0])
        initial_state.set_thetadot(start_state[1])
        dircol.prog().AddBoundingBoxConstraint(
            initial_state.get_value(), initial_state.get_value(), dircol.initial_state()
        )

        final_state = PendulumState()
        final_state.set_theta(goal_state[0])
        final_state.set_thetadot(goal_state[1])
        dircol.prog().AddBoundingBoxConstraint(
            final_state.get_value(), final_state.get_value(), dircol.final_state()
        )

        dircol.AddRunningCost(control_cost * u[0] ** 2)

        if initial_x_trajectory is not None:
            dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

        result = Solve(dircol.prog())
        assert result.is_success()

        x_trajectory = dircol.ReconstructStateTrajectory(result)
        return x_trajectory, dircol, result

    def plot_phase_space_trajectory(self, x_trajectory, save_to=None):
        """
        Plot the computed trajectory in phase space.

        Parameters
        ----------
        x_trajectory : pydrake.trajectories.PiecewisePolynomial
            the trajectory returned from the compute_trajectory function.
        save_to : string, default=None
            string pointing to the location where the figure is supposed
            to be stored. If save_to==None, the figure is not stored but shown
            in a window instead.
        """
        fig, ax = plt.subplots()

        time = np.linspace(x_trajectory.start_time(), x_trajectory.end_time(), 100)

        x_knots = np.hstack([x_trajectory.value(t) for t in time])

        ax.plot(x_knots[0, :], x_knots[1, :])
        if save_to is None:
            plt.show()
        else:
            plt.xlim(-np.pi, np.pi)
            plt.ylim(-10, 10)
            plt.savefig(save_to)
        plt.close()

    def extract_trajectory(self, x_trajectory, dircol, result, N=801):
        """
        Extract time, position, velocity and control trajectories from
        the outputs of the compute_trajectory function.

        Parameters
        ----------
        x_trajectory : pydrake.trajectories.PiecewisePolynomial
            trajectory in state space
        dircol : pydrake.systems.trajectory_optimization.DirectCollocation
            DirectCollocation pydrake object
        result : pydrake.solvers.mathematicalprogram.MathematicalProgramResult
            MathematicalProgramResult pydrake object
        N : int, default=801
            The number of sampling points of the returned trajectories

        Returns
        -------
        time_traj : array_like
            the time trajectory
        theta : array_like
            the position trajectory
        theta_dot : array_like
            the velocity trajectory
        torque_traj : array_like
            the control (torque) trajectory
        """
        # Extract Time
        time = np.linspace(x_trajectory.start_time(), x_trajectory.end_time(), N)
        T = time.reshape(N, 1).T[0]

        # Extract State
        X = np.hstack([x_trajectory.value(t) for t in time]).T

        # Extract Control Inputs
        u_trajectory = dircol.ReconstructInputTrajectory(result)
        U = np.hstack([u_trajectory.value(t) for t in time])[0]

        return T, X, U
