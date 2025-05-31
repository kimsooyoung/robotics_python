# referenced from : https://github.com/dfki-ric-underactuated-lab/torque_limited_simple_pendulum
import sys, os

import numpy as np

from functools import partial
from matplotlib import pyplot as plt

from pendulum_env import (
    # env
    Simulator,
    PendulumPlant,
    # controller
    iLQRMPCController
)
from utils import (
    plot_trajectory,
)

log_dir = "log_data/ilqr"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

class Parameters():

    def __init__(self):
        # pendulum parameters
        self.m = 1.0 # mass
        self.l = 0.5 # length
        self.c = 0.0 # coulomb friction coefficient
        self.b = 0.1 # damping friction coefficient
        self.I = self.m * self.l * self.l # inertia
        self.g = 9.81 # gravity
        self.pause = 0.02
        self.fps = 20

        # swingup parameters
        # self.dt = 0.01
        self.dt = 0.02
        self.t_final = 10.0
        self.x0 = [0.0, 0.0]
        self.goal = [np.pi, 0.0]
        self.torque_limit = 5
        # self.torque_limit = 50

        # iLQR parameters
        self.N = 50 # horizon size
        self.max_iter = 1
        self.n_x = 2
        self.n_u = 1
        self.integrator = "runge_kutta"

        # cost function weights
        """
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
        """
        self.sCu = 30.0
        self.sCp = 10.0
        self.sCv = 1.0
        self.sCen = 1.0
        self.fCp = 10.0
        self.fCv = 1.0
        self.fCen = 80.0


if __name__ == '__main__':

    params = Parameters()

    # Simulate trajectory
    pendulum = PendulumPlant(
        mass=params.m,
        length=params.l,
        damping=params.b,
        gravity=params.g,
        coulomb_fric=params.c,
        inertia=None,
        torque_limit=params.torque_limit,
    )
    sim = Simulator(plant=pendulum)

    controller = iLQRMPCController(
        mass=params.m, length=params.l,
        damping=params.b, coulomb_friction=params.c,
        gravity=params.g, inertia=params.I,
        dt=params.dt, n=params.N, max_iter=params.max_iter,
        break_cost_redu=1e-1,
        sCu=params.sCu, sCp=params.sCp, 
        sCv=params.sCv, sCen=params.sCen,
        fCp=params.fCp, fCv=params.fCv, fCen=params.fCen,
        dynamics=params.integrator, n_x=params.n_x
    )

    x0_sim = params.x0.copy()
    controller.set_goal(params.goal)
    controller.init(x0=x0_sim)

    T, X, U = sim.simulate_and_animate(
        t0=0.0, x0=x0_sim, tf=params.t_final,
        dt=params.dt, controller=controller, 
        integrator=params.integrator
    )
    
    plot_trajectory(T, X, U, None, True)