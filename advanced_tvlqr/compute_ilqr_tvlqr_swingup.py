# referenced from : https://github.com/dfki-ric-underactuated-lab/torque_limited_simple_pendulum
import sys, os

import numpy as np

from functools import partial
from matplotlib import pyplot as plt

from pendulum_env import (
    # env
    Simulator,
    PendulumPlant,
    # trajectory optimization
    iLQR_Calculator,
    # sympy dynamics
    pendulum_discrete_dynamics_rungekutta,
    # sympy cost functions
    pendulum_swingup_stage_cost,
    pendulum_swingup_final_cost,
    # controller
    PIDController,
    LQRController,
    TVLQRController,
)
from utils import (
    prepare_empty_data_dict, 
    plot_trajectory,
    save_trajectory,
    plot_ilqr_trace
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

        self.n_x = 2
        self.n_u = 1
        self.integrator = "runge_kutta"

        # swingup parameters
        self.dt = 0.01
        self.x0 = [0.0, 0.0]
        self.goal = [np.pi, 0.0]
        # self.torque_limit = 5
        self.torque_limit = 50

        # iLQR parameters
        self.N = 1000
        self.max_iter = 100
        self.regu_init = 100
        # cost function weights
        self.sCu = 0.1
        self.sCp = 0.0
        self.sCv = 0.0
        self.sCen = 0.0
        self.fCp = 1000.0
        self.fCv = 1.0
        self.fCen = 0.0

        # # pendulum parameters
        # mass = 0.57288
        # length = 0.5
        # damping = 0.10
        # gravity = 9.81
        # coulomb_fric = 0.0
        # torque_limit = 1.5

        # self.m = 0.3 # mass
        # self.l = 1.0 # length
        # self.c = 0.0 # coulomb friction coefficient
        # self.b = 0.1 # damping friction coefficient
        # self.I = self.m * self.l * self.l # inertia
        # self.g = 9.81 # gravity
        # self.pause = 0.02
        # self.fps = 20


if __name__ == '__main__':

    params = Parameters()

    # Compute trajectory
    iLQR = iLQR_Calculator(n_x=params.n_x, n_u=params.n_u)
    dyn_func = pendulum_discrete_dynamics_rungekutta

    # set dynamics
    dyn = partial(
        dyn_func, 
        dt=params.dt,
        m=params.m, 
        l=params.l, 
        b=params.b, 
        cf=params.c, 
        g=params.g, 
        inertia=params.I
    )
    iLQR.set_discrete_dynamics(dyn)

    # set costs
    s_cost_func = pendulum_swingup_stage_cost
    f_cost_func = pendulum_swingup_final_cost
    s_cost = partial(
        s_cost_func,
        goal=params.goal,
        Cu=params.sCu,
        Cp=params.sCp,
        Cv=params.sCv,
        Cen=params.sCen,
        m=params.m,
        l=params.l,
        b=params.b,
        cf=params.c,
        g=params.g
    )
    f_cost = partial(
        f_cost_func,
        goal=params.goal,
        Cp=params.fCp,
        Cv=params.fCv,
        Cen=params.fCen,
        m=params.m,
        l=params.l,
        b=params.b,
        cf=params.c,
        g=params.g
    )
    iLQR.set_stage_cost(s_cost)
    iLQR.set_final_cost(f_cost)

    iLQR.init_derivatives()
    iLQR.set_start(params.x0)

    # computation
    (x_trj, u_trj, cost_trace, regu_trace,
    redu_ratio_trace, redu_trace) = iLQR.run_ilqr(
        N=params.N,
        init_u_trj=None,
        init_x_trj=None,
        max_iter=params.max_iter,
        regu_init=params.regu_init,
        break_cost_redu=1e-6
    )
    # preprocess results
    time = np.linspace(0, params.N-1, params.N)*params.dt
    TH = x_trj.T[0]
    THD = x_trj.T[1]

    data_dict = prepare_empty_data_dict(params.dt, params.N*params.dt)
    data_dict["des_time"] = time
    data_dict["des_pos"] = TH
    data_dict["des_vel"] = THD
    data_dict["des_tau"] = np.append(u_trj.T[0], 0.0)
    
    csv_path = os.path.join(log_dir, "computed_trajectory.csv")
    save_trajectory(csv_path, data_dict)

    # # plot results
    # plot_trajectory(time, x_trj, data_dict["des_tau"], None, True)
    # plot_ilqr_trace(cost_trace, redu_ratio_trace, regu_trace)

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

    # controller = PIDController(
    #     data_dict=data_dict, 
    #     Kp=20.0, 
    #     Ki=1.0,
    #     Kd=1.0
    # )
    controller = TVLQRController(
        data_dict=data_dict,
        mass=params.m,
        length=params.l,
        damping=params.b,
        gravity=params.g,
        torque_limit=params.torque_limit,
    )
    # controller = OpenLoopController(data_dict=data_dict)

    controller.set_goal(params.goal)

    dt = data_dict["des_time"][1] - data_dict["des_time"][0]
    t_final = data_dict["des_time"][-1]

    T, X, U = sim.simulate_and_animate(
        t0=0.0,
        x0=params.x0,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator="runge_kutta",
        phase_plot=False,
        save_video=False,
    )

    plot_trajectory(T, X, U, None, True)
