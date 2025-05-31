"""
iLQR MPC Controller
===================
"""


# Other imports
import numpy as np
from functools import partial
try:
    import pydrake
    pydrake_available = True
except ModuleNotFoundError:
    pydrake_available = False

# Local imports
# TODO: sympy iLQR
# if pydrake_available:
#     from simple_pendulum.trajectory_optimization.ilqr.ilqr import iLQR_Calculator
# else:
#     from simple_pendulum.trajectory_optimization.ilqr.ilqr_sympy import iLQR_Calculator

from .ilqr import iLQR_Calculator
from .pendulum_sympy import (
    pendulum_discrete_dynamics_euler,
    pendulum_discrete_dynamics_rungekutta,
    pendulum_swingup_stage_cost,
    pendulum_swingup_final_cost,
    pendulum3_discrete_dynamics_euler,
    pendulum3_discrete_dynamics_rungekutta,
    pendulum3_swingup_stage_cost,
    pendulum3_swingup_final_cost
)

class iLQRMPCController:
    """
    Controller which computes an ilqr solution at every timestep and uses
    the first control output.
    """
    def __init__(self,
                 mass=0.5,
                 length=0.5,
                 damping=0.15,
                 coulomb_friction=0.0,
                 gravity=9.81,
                 inertia=0.125,
                 torque_limit=2.0,
                 dt=0.01,
                 n=50,
                 max_iter=1,
                 break_cost_redu=1e-6,
                 sCu=10.0,
                 sCp=0.001,
                 sCv=0.001,
                 sCen=0.0,
                 fCp=1000.0,
                 fCv=10.0,
                 fCen=300.0,
                 dynamics="runge_kutta",
                 n_x=3):
        """
        Controller which computes an ilqr solution at every timestep and uses
        the first control output.

        Parameters
        ----------
        mass : float, default=1.0
            mass of the pendulum [kg]
        length : float, default=0.5
            length of the pendulum [m]
        damping : float, default=0.1
            damping factor of the pendulum [kg m/s]
        coulomb_friction : float, default=0.0
            coulomb_friciton term of the pendulum
        gravity : float, default=9.81
            gravity (positive direction points down) [m/s^2]
        inertia : float, default=0.125
            inertia of the pendulum
        torque_limit : float, default=2.0
            torque limit of the motor [Nm]
        dt : float, default=0.01
            timestep of the simulation
        n : int, default=50
            number of timnesteps the controller optimizes ahead
        max_iter : int, default=1
            optimization iterations the alogrithm makes at every timestep
        break_cost_redu : float, default=1e-6
            cost at which the optimization breaks off early
        sCu : float, default=10.0
            running cost weight for the control input u
        sCp : float, default=0.001
            running cost weight for the position error
        sCv : float, default=0.001
            running cost weight for the velocity error
        sCen : float, default=0.0
            running cost weight for the energy error
        fCp : float, default=1000.0
            final cost weight for the position error
        fCv : float, default=10.0
            final cost weight for the velocity error
        fCen : float, default=300.0
            final cost weight for the energy error
        dynamics : string, default="runge_kutta"
            string that selects the integrator to be used for the simulation
            options are: "euler", "runge_kutta"
        n_x : int, default=3
            determines how the state space of the pendulum is represented
            n_x=2 means state = [position, velocity]
            n_x=3 means state = [cos(position), sin(position), velocity]
        """

        self.mass = mass
        self.length = length
        self.damping = damping
        self.coulomb_friction = coulomb_friction
        self.gravity = gravity
        self.torque_limit = torque_limit

        self.N = n
        self.n_x = n_x

        self.sCu = sCu
        self.sCp = sCp
        self.sCv = sCv
        self.sCen = sCen
        self.fCp = fCp
        self.fCv = fCv
        self.fCen = fCen

        self.break_cost_redu = break_cost_redu
        self.max_iter = max_iter

        # Setup dynamics function in ilqr calculator
        self.iLQR = iLQR_Calculator(n_x=n_x, n_u=1)
        if n_x == 2:
            if dynamics == "euler":
                dyn_func = pendulum_discrete_dynamics_euler
            else:
                dyn_func = pendulum_discrete_dynamics_rungekutta
        elif n_x == 3:
            if dynamics == "euler":
                dyn_func = pendulum3_discrete_dynamics_euler
            else:
                dyn_func = pendulum3_discrete_dynamics_rungekutta
        dyn = partial(dyn_func,
                      dt=dt,
                      m=mass,
                      l=length,
                      b=damping,
                      cf=coulomb_friction,
                      g=gravity,
                      inertia=inertia)
        self.iLQR.set_discrete_dynamics(dyn)

        # set default start position
        if self.n_x == 2:
            x = np.array([0.0, 0.0])
        elif self.n_x == 3:
            x = np.array([1.0, 0.0, 0.0])
        self.iLQR.set_start(x)

    def init(self, x0):
        if self.n_x == 2:
            x = np.copy(x0)
        elif self.n_x == 3:
            x = np.array([np.cos(x0[0]), np.sin(x0[0]), x0[1]])
        self.iLQR.set_start(x)
        self.compute_initial_guess(verbose=False)

    def load_initial_guess(self, filepath="Pendulum_data/trajectory.csv",
                           verbose=True):
        '''
        load initial guess trajectory from file

        Parameters
        ----------
        filepath : string, default="Pendulum_data/trajectory.csv"
            path to the csv file containing the initial guess for
            the trajectory
        verbose : bool, default=True
            whether to print from where the initial guess is loaded
        '''
        if verbose:
            print("Loading initial guess from ", filepath)
        if self.n_x == 2:
            trajectory = np.loadtxt(filepath, skiprows=1, delimiter=",")
            self.u_trj = trajectory.T[3].T[:self.N]
            self.u_trj = np.expand_dims(self.u_trj, axis=1)
            self.x_trj = trajectory.T[1:3].T[:self.N]
            self.x_trj = np.insert(self.x_trj, 0, self.x_trj[0], axis=0)
        if self.n_x == 3:
            trajectory = np.loadtxt(filepath, skiprows=1, delimiter=",")
            self.u_trj = trajectory.T[3].T[:self.N]
            self.u_trj = np.expand_dims(self.u_trj, axis=1)
            self.x_trj = trajectory.T[1:3].T[:self.N]
            self.x_trj = np.insert(self.x_trj, 0, self.x_trj[0], axis=0)
            sin = np.sin(self.x_trj.T[0])
            cos = np.cos(self.x_trj.T[0])
            v = self.x_trj.T[1]
            self.x_trj = np.stack((cos, sin, v), axis=1)

    def set_initial_guess(self, u_trj=None, x_trj=None):
        '''
        set initial guess from array like object

        Parameters
        ----------
        u_trj : array-like, default=None
            initial guess for control inputs u
            ignored if u_trj==None
        x_trj : array-like, default=None
            initial guess for state space trajectory
            ignored if x_trj==None
        '''
        if u_trj is not None:
            self.u_trj = u_trj[:self.N]
        if x_trj is not None:
            self.x_trj = x_trj[:self.N]

    def compute_initial_guess(self, N=None, verbose=True):
        '''
        compute initial guess

        Parameters
        ----------
        N : int, default=None
            number of timesteps to plan ahead
            if N==None, N defaults to the number of timesteps that is also
            used during the online optimization (n in the class __init__)
        verbose : bool, default=True
            whether to print when the initial guess calculation is finished
        '''
        if verbose:
            print("Computing initial guess")
        if N is None:
            N = self.N

        (self.x_trj, self.u_trj,
         cost_trace, regu_trace,
         redu_ratio_trace,
         redu_trace) = self.iLQR.run_ilqr(N=N,
                                          init_u_trj=None,
                                          init_x_trj=None,
                                          max_iter=500,
                                          regu_init=100,
                                          break_cost_redu=1e-6)
        self.x_trj = self.x_trj[:self.N]
        self.u_trj = self.u_trj[:self.N]
        if verbose:
            print("Computing initial guess done")

    def set_goal(self, x):
        """
        Set a goal for the controller. Initializes the cost functions.

        Parameters
        ----------
        x : array-like
            goal state for the pendulum
        """
        if self.n_x == 2:
            s_cost_func = pendulum_swingup_stage_cost
            f_cost_func = pendulum_swingup_final_cost
            g = np.copy(x)
        elif self.n_x == 3:
            s_cost_func = pendulum3_swingup_stage_cost
            f_cost_func = pendulum3_swingup_final_cost
            g = np.array([np.cos(x[0]), np.sin(x[0]), x[1]])

        s_cost = partial(s_cost_func,
                         goal=g,
                         Cu=self.sCu,
                         Cp=self.sCp,
                         Cv=self.sCv,
                         Cen=self.sCen,
                         m=self.mass,
                         l=self.length,
                         b=self.damping,
                         cf=self.coulomb_friction,
                         g=self.gravity)
        f_cost = partial(f_cost_func,
                         goal=g,
                         Cp=self.fCp,
                         Cv=self.fCv,
                         Cen=self.fCen,
                         m=self.mass,
                         l=self.length,
                         b=self.damping,
                         cf=self.coulomb_friction,
                         g=self.gravity)

        self.iLQR.set_stage_cost(s_cost)
        self.iLQR.set_final_cost(f_cost)
        self.iLQR.init_derivatives()

    def get_control_output(self, meas_pos, meas_vel,
                           meas_tau=0, meas_time=0):
        """
        The function to compute the control input for the pendulum actuator

        Parameters
        ----------
        meas_pos : float
            the position of the pendulum [rad]
        meas_vel : float
            the velocity of the pendulum [rad/s]
        meas_tau : float, default=0
            the meastured torque of the pendulum [Nm]
            (not used)
        meas_time : float, default=0
            the collapsed time [s]
            (not used)

        Returns
        -------
        des_pos : float
            the desired position of the pendulum [rad]
            (not used, returns None)
        des_vel : float
            the desired velocity of the pendulum [rad/s]
            (not used, returns None)
        des_tau : float
            the torque supposed to be applied by the actuator [Nm]
        """
        pos = float(np.squeeze(meas_pos))
        vel = float(np.squeeze(meas_vel))
        pos = pos % (2*np.pi)

        if self.n_x == 2:
            state = np.array([pos, vel])
        if self.n_x == 3:
            state = np.asarray([np.cos(pos),
                                np.sin(pos),
                                vel])
        self.iLQR.set_start(state)
        (self.x_trj, self.u_trj,
         cost_trace, regu_trace,
         redu_ratio_trace,
         redu_trace) = self.iLQR.run_ilqr(
             init_u_trj=self.u_trj,
            init_x_trj=None,
            shift=True,
            max_iter=self.max_iter,
            regu_init=100,
            break_cost_redu=self.break_cost_redu
        )

        # since this is a pure torque controller,
        # set pos_des and vel_des to None
        des_pos = None
        des_vel = None
        des_tau = min(self.u_trj[0], self.torque_limit)
        des_tau = max(des_tau, -self.torque_limit)

        return des_pos, des_vel, float(des_tau)
