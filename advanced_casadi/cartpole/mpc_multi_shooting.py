# ref from : https://github.com/tomcattiger1230/CasADi_MPC_MHE_Python/tree/master

import casadi as ca
import casadi.tools as ca_tools

import time
import numpy as np
from draw import animate, plot

class Param:

    def __init__(self):
        self.g = 9.81
        self.M = 0.5
        self.m = 0.2
        self.L = 0.3
        self.I = 0.006
        self.d = 0.1

        # TODO: Tune weight matrices
        # self.Q = np.diag([1., 1., 1., 1.])
        self.Q = np.diag([1., 1., 10., 1.])
        self.R = np.diag([0.01])

def shift_movement(T, t0, x0, u, dynamics_func):

    f_value = dynamics_func(x0, u[:, 0])
    st = x0 + T * f_value
    t = t0 + T

    # append last command to the end
    u_end = ca.horzcat(u[:, 1:], u[:, -1])

    return t, st, u_end.T

if __name__ == '__main__':

    # ## Case 1
    # T = 0.1 # sampling time [s]
    # N = 20 # prediction horizon

    # ## Case 2
    # T = 0.1 # sampling time [s]
    # N = 40 # prediction horizon
    
    ## Case 3
    T = 0.1 # sampling time [s]
    N = 100 # prediction horizon

    # ## Case 4
    # T = 0.1 # sampling time [s]
    # N = 250 # prediction horizon

    param = Param()
    g, m, M, L, I, d = param.g, param.m, param.M, param.L, param.I, param.d

    # max values for states and controls
    u_max = 200.0
    x_max = 4.0
    v_max = 10.0
    theta_max = np.pi
    w_max = 10.0

    # Declare model variables
    x, x_dot = ca.SX.sym('x'), ca.SX.sym('x_dot')
    theta, theta_dot = ca.SX.sym('theta'), ca.SX.sym('theta_dot')
    u = ca.SX.sym('u')
    X = ca.vertcat(x, x_dot, theta, theta_dot)
    n_states = X.size()[0]

    # derivitive of the X
    ax = 1.0*(2.0*L*m*theta_dot**2*np.sin(theta) - 2.0*d*x_dot + g*m*np.sin(2*theta)/2 + 2.0*u)/(2*M + m*np.sin(theta)**2 + m)
    alpha = 1.0*(-g*(M + m)*np.sin(theta) - (1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + u)*np.cos(theta))/(L*(2*M + m*np.sin(theta)**2 + m))

    X_dot = ca.vertcat(x_dot, ax, theta_dot, alpha)

    # control variables
    U = ca.vertcat(u)
    n_controls = U.size()[0]

    ## function
    dynamics_func = ca.Function('dynamics_func', [X, U], [X_dot], ['input_state', 'control_input'], ['rhs'])

    ## for MPC

    # N Control Variables 
    U = ca.SX.sym('U', n_controls, N)
    # N + 1 State variables
    X = ca.SX.sym('X', n_states, N+1)
    # Initial and Final states (MPC optimization parameters)
    P = ca.SX.sym('P', n_states + n_states)
    # weighing matrices (states)
    Q = param.Q
    # weighing matrices (controls)
    R = param.R

    # Objective function
    obj = 0
    # constraints - x/y states, initial state, dynamics contraints
    g = []
    # initial state
    g.append(X[:, 0]-P[:4])

    # Append the rest of the dynamics constraints and MPC cost function
    for i in range(N):
        obj = obj + (X[:4, i] - P[4:8]).T @ Q @ (X[:4, i] - P[4:8]) + U[:, i].T @ R @ U[:, i]
        x_next = dynamics_func(X[:, i], U[:, i]) * T + X[:, i]
        g.append(X[:, i+1]-x_next)

        # obj = obj + (X[:4, i] - P[3:6]).T @ Q @ (X[:3, i] - P[3:6]) + U[:, i].T @ R @ U[:, i]
        # x_next = dynamics_func(X[:, i], U[:, i]) * T + X[:, i]
        # g.append(X[:, i+1]-x_next)

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

    # Control Bounds
    lbx = []
    ubx = []

    # first, state constraint bounds
    for _ in range(N):
        lbx = ca.vertcat(lbx, -u_max)
        ubx = ca.vertcat(ubx, u_max)
    # second, control constraint bounds
    for _ in range(N+1):
        lbx = ca.vertcat(lbx, -x_max, -v_max, -theta_max, -w_max)
        ubx = ca.vertcat(ubx, x_max, v_max, theta_max, w_max)

    # Constraint Bounds
    lbg = 0.0
    ubg = 0.0

    # Simulation
    t0 = 0.0
    x0 = np.array([0, 0, np.pi+0.1, 0]).reshape(-1, 1) # initial state
    xs = np.array([1, 0, np.pi, 0]).reshape(-1, 1) # final state
    u0 = np.zeros((N, n_controls)) # controls
    
    # MPC opt variables
    x_m = np.zeros((n_states, N+1))
    init_control = np.concatenate((u0.reshape(-1, 1), x_m.reshape(-1, 1)))

    # solver param - initial and final state
    c_p = np.concatenate((x0, xs))

    ###############
    ## start MPC ##
    ###############

    sim_time = 30.0
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
        result_u.append(u0[0].full())
        time_list.append(t0)

        # update initial state
        mpciter = mpciter + 1

    time_list = np.array(time_list)

    print(f"Average solve time: {time_list.mean()}")
    print(f"Minimum solve time: {time_list.min()}")

    # result reshape for animation
    result_x = np.array(result_x).reshape(-1, n_states)
    result_u = np.array(result_u).reshape(-1, n_controls)
    
    # visualization
    animate(time_list, result_x, L, T/4)
    plot(time_list, result_x, result_u)