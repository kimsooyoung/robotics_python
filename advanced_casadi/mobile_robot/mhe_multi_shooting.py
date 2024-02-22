# ref from : https://github.com/tomcattiger1230/CasADi_MPC_MHE_Python/tree/master

import casadi as ca
import casadi.tools as ca_tools

import time
import numpy as np
from draw import Draw_MPC_point_stabilization_v1, draw_gt, draw_gt_measurements, draw_gtmeas_noisemeas, draw_gt_mhe_measurements

# MHE (Moving Horizon Estimation)
# Unlike deterministic approaches, MHE requires an iterative approach that
# relies on linear programming or nonlinear programming solvers to find a solution.

def shift_movement(T, t0, x0, u, dynamics_func):

    control_cov = np.diag([0.005, np.deg2rad(2)])**2
    control = u[:, 0] + np.sqrt(control_cov) @ np.random.randn(2, 1)

    f_value = dynamics_func(x0, control)
    st = x0 + T * f_value
    t = t0 + T

    # append last command to the end
    # u_end = ca.horzcat(u[:, 1:], u[:, -1])
    u_end = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)

    return t, st, u_end

if __name__ == '__main__':

    ## Case 1
    xs = [1.5, 1.5, 0.0] # final state
    T = 0.2 # sampling time [s]
    N = 10 # prediction horizon

    # ## Case 2
    # xs = [1.5, 1.5, np.pi] # final state
    # T = 0.2 # sampling time [s]
    # N = 10 # prediction horizon
    
    # ## Case 3
    # xs = [1.5, 1.5, 0.0] # final state
    # T = 0.2 # sampling time [s]
    # N = 25 # prediction horizon

    # ## Case 4
    # xs = [1.5, 1.5, 0.0] # final state
    # T = 0.2 # sampling time [s]
    # N = 100 # prediction horizon    

    rob_diam = 0.3 # [m]
    v_max = 0.6
    omega_max = np.pi / 4.0

    # Declare model variables
    x, y, theta = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('theta')
    states = ca.vertcat(x, y, theta)
    n_states = states.size()[0]

    # control variables
    v, omega = ca.SX.sym('v'), ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    n_controls = controls.size()[0]

    ## rhs
    rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)

    ## function
    dynamics_func = ca.Function('dynamics_func', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC

    # N Control Variables 
    U = ca.SX.sym('U', n_controls, N)
    # N + 1 State variables
    X = ca.SX.sym('X', n_states, N+1)
    # Initial and Final states (MPC optimization parameters)
    P = ca.SX.sym('P', n_states + n_states)
    # weighing matrices (states)
    Q = np.diag([1, 5, 0.1])
    # weighing matrices (controls)
    R = np.diag([0.5, 0.05])

    # Objective function
    obj = 0
    # constraints - x/y states, initial state, dynamics contraints
    g = []
    # initial state
    g.append(X[:, 0]-P[:3])

    # Append the rest of the dynamics constraints and MPC cost function
    for i in range(N):
        obj = obj + (X[:3, i] - P[3:6]).T @ Q @ (X[:3, i] - P[3:6]) + U[:, i].T @ R @ U[:, i]
        x_next = dynamics_func(X[:, i], U[:, i]) * T + X[:, i]
        g.append(X[:, i+1]-x_next)

    opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

    nlp_prob = {
        'f': obj, 
        'x': opt_variables, 
        'p': P,
        'g': ca.vertcat(*g),
    }
    opts_setting = {
        'ipopt.max_iter': 100, 
        'ipopt.print_level': 0,
        'print_time': 0, 
        'ipopt.acceptable_tol': 1e-8, 
        'ipopt.acceptable_obj_change_tol': 1e-6
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    # THE SIMULATION LOOP SHOULD START FROM HERE

    # State Bounds
    lbx = []
    ubx = []

    # first, state constraint bounds
    for _ in range(N):
        lbx = ca.vertcat(lbx, -v_max, -omega_max)
        ubx = ca.vertcat(ubx, v_max, omega_max)
    # second, control constraint bounds
    for _ in range(N+1):
        lbx = ca.vertcat(lbx, -2.0, -2.0, -np.inf)
        ubx = ca.vertcat(ubx, 2.0, 2.0, np.inf)

    # Constraint Bounds
    lbg = 0.0
    ubg = 0.0

    # Simulation
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1) # initial state
    xs = np.array([xs]).reshape(-1, 1) # final state
    u0 = np.zeros((N, n_controls)) # controls
    
    # MPC opt variables
    x_m = np.zeros((n_states, N+1))
    init_control = np.concatenate((u0.reshape(-1, 1), x_m.reshape(-1, 1)))

    # solver param - initial and final state
    c_p = np.concatenate((x0, xs))

    ###############
    ## start MPC ##
    ###############

    sim_time = 20.0
    mpciter = 0

    # store data
    result_x = []
    predict_x = []
    result_u = []
    time_list = []
    calc_time_list = []

    start_time = time.time()
    
    # the main simulaton loop... it works as long as the error is greater
    # than 10^-2 and the number of mpc steps is less than its maximum value.
    while(np.linalg.norm(x0-xs) > 1e-2 and mpciter - (sim_time / T) < 0.0):
        
        # set the values of the parameters vector
        c_p = np.concatenate((x0, xs))

        # initial value of the optimization variables
        init_control = np.concatenate((ca.reshape(u0, -1, 1), ca.reshape(x_m, -1, 1)))

        # call the optimizer
        solve_tic = time.time()
        sol = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        solve_toc = time.time() - solve_tic
        calc_time_list.append(solve_toc)

        # Current X contains the state variables and control variables
        opt_variables = sol['x'].full()
        u_sol = ca.reshape(opt_variables[:n_controls*N], (n_controls, N))
        x_m = ca.reshape(opt_variables[n_controls*N:], (n_states, N+1))

        # list of predicted states - [n_states, N+1]
        predict_x.append(x_m.full())
        
        # dynamics step forward
        t0, x0, u0 = shift_movement(T, t0, x0, u_sol, dynamics_func)

        # store the results
        result_x.append(x0.full())
        result_u.append(u0[:, 0])
        time_list.append(t0)

        # update initial state
        mpciter = mpciter + 1
        # break

    # synthesize the measurements
    control_cov = np.diag([0.005, np.deg2rad(2)])**2
    measurement_cov = np.diag([0.1, np.deg2rad(2)])**2


    r = []
    alpha = []
    for i in range(len(time_list)):
        r.append(
            np.sqrt(
                result_x[i][0] ** 2 + result_x[i][1] ** 2
            ) + np.sqrt(measurement_cov[0, 0]) * np.random.randn()
        )
        alpha.append(
            np.arctan(
                result_x[i][1] / result_x[i][0]
            ) + np.sqrt(measurement_cov[1, 1]) * np.random.randn()
        )
    y_measurements = np.concatenate(
        (
            np.array(r).reshape(-1, 1), 
            np.array(alpha).reshape(-1, 1)
        ), axis=1
    )

    # Draw the ground truth and the measurements
    # draw_gt_measurements(time_list, result_x, y_measurements)

    ##################
    ## Offline MHE ###
    ##################

    # MHE variables
    T_mhe = 0.2
    # estimation horizon 
    # N_mhe = 6 
    N_mhe = len(time_list) - 1

    # measurement model
    r = ca.SX.sym('r')
    alpha = ca.SX.sym('alpha')
    measurement_rhs = ca.vertcat(
        ca.sqrt(x**2 + y**2), 
        ca.atan(y/x)
    )
    measurement_func = ca.Function('measurement_func', [states], [measurement_rhs], ['states'], ['meas'])

    # using the same model and states （states, dynamics_func, next_controls)
    mhe_U = ca.SX.sym('mhe_U', n_controls, N_mhe)
    mhe_X = ca.SX.sym('mhe_X', n_states, N_mhe+1)

    _, n_mes = y_measurements.shape
    Mes_ref = ca.SX.sym('Mes_ref', n_mes, N_mhe+1)
    U_ref = ca.SX.sym('U_ref', n_controls, N_mhe)

    # weight matrices
    V_mat = np.linalg.inv(np.sqrt(measurement_cov))
    W_mat = np.linalg.inv(np.sqrt(control_cov))

    # Objective function
    mhe_obj = 0
    for i in range(N_mhe+1):
        h_x = measurement_func(mhe_X[:, i])
        # measured output - estimated output
        temp_diff_ = Mes_ref[:, i] - h_x
        mhe_obj = mhe_obj + ca.mtimes([temp_diff_.T, V_mat, temp_diff_])
    for i in range(N_mhe):
        temp_diff_ = U_ref[:, i] - mhe_U[:, i]
        mhe_obj = mhe_obj + ca.mtimes([temp_diff_.T, W_mat, temp_diff_])

    # constraints
    g = [] # equal constrains
    # multiple shooting constraints
    for i in range(N_mhe):
        x_next_ = dynamics_func(mhe_X[:, i], mhe_U[:, i])*T_mhe + mhe_X[:, i]
        g.append(mhe_X[:, i+1] - x_next_)

    mhe_target = ca.vertcat(ca.reshape(mhe_U, -1, 1), ca.reshape(mhe_X, -1, 1))
    mhe_params = ca.vertcat(ca.reshape(U_ref, -1, 1), ca.reshape(Mes_ref, -1, 1))
    
    ### define MHE nlp problem, the constraints stay the same as MPC
    nlp_prob_mhe = {
        'f': mhe_obj, 
        'x': mhe_target, 
        'p':mhe_params, 
        'g':ca.vertcat(*g)
    }
    mhe_opts_setting = {
        'ipopt.max_iter':2000, 
        'ipopt.print_level':0, 
        'print_time':0, 
        'ipopt.acceptable_tol':1e-8, 
        'ipopt.acceptable_obj_change_tol':1e-6
    }
    mhe_solver = ca.nlpsol('solver', 'ipopt', nlp_prob_mhe, mhe_opts_setting)

    # State Bounds
    mhe_lbx = []
    mhe_ubx = []

    # first, state constraint bounds
    for _ in range(N_mhe):
        mhe_lbx = ca.vertcat(mhe_lbx, -v_max, -omega_max)
        mhe_ubx = ca.vertcat(mhe_ubx, v_max, omega_max)
    # second, control constraint bounds
    for _ in range(N_mhe+1):
        mhe_lbx = ca.vertcat(mhe_lbx, -2.0, -2.0, -np.inf)
        mhe_ubx = ca.vertcat(mhe_ubx, 2.0, 2.0, np.inf)

    # Constraint Bounds
    mhe_lbg = 0.0
    mhe_ubg = 0.0
    
    # initial state and control
    x0_mhe = np.zeros((N_mhe+1, n_states))
    for i in range(N_mhe+1):
        x0_mhe[i] = np.array([
            y_measurements[i, 0] * np.cos(y_measurements[i, 1]),
            y_measurements[i, 0] * np.sin(y_measurements[i, 1]),
            0.0
        ])
    u0_mhe = np.array(result_u[:N_mhe])
    # x0_mhe: (7, 3)
    # u0_mhe: (6, 2)

    # store data
    result_x_mhe = []
    result_u_mhe = []

    # initial state and control
    init_control_mhe = np.concatenate((u0_mhe.reshape(-1, 1), x0_mhe.reshape(-1, 1)))

    for i in range(y_measurements.shape[0]-N_mhe):
        mhe_c_p = np.concatenate((
            np.array(result_u[i:i+N_mhe]).reshape(-1, 1),
            y_measurements[i:i+N_mhe+1].reshape(-1, 1))
        )

        mhe_res = mhe_solver(x0=init_control_mhe, p=mhe_c_p, lbg=mhe_lbg, lbx=mhe_lbx, ubg=mhe_ubg, ubx=mhe_ubx)
        mhe_estimated = mhe_res['x'].full()

        mhe_u_sol = mhe_estimated[:n_controls*N_mhe].reshape(N_mhe, n_controls)
        mhe_state_sol = mhe_estimated[n_controls*N_mhe:].reshape(N_mhe+1, n_states)

        result_u_mhe.append(mhe_u_sol[N_mhe-1:])
        result_x_mhe.append(mhe_state_sol[N_mhe:])

        x0_mhe = np.concatenate((mhe_state_sol[1:], mhe_state_sol[-1:]))
        u0_mhe = np.concatenate((mhe_u_sol[1:], mhe_u_sol[-1:]))

    draw_gt_mhe_measurements(time_list, result_x, y_measurements, result_x_mhe, n_mhe=N_mhe)
    