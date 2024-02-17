# ref from : https://github.com/tomcattiger1230/CasADi_MPC_MHE_Python/tree/master

import casadi as ca
import casadi.tools as ca_tools

import time
import numpy as np
from draw import Draw_MPC_Obstacle

# Multi Shooting MPC means
# 1. state variables as opt variables
# 2. dynamics as MPC constraints

def shift_movement(T, t0, x0, u, dynamics_func):

    f_value = dynamics_func(x0, u[:, 0])
    st = x0 + T * f_value
    t = t0 + T

    # append last command to the end
    u_end = ca.horzcat(u[:, 1:], u[:, -1])

    return t, st, u_end.T

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

    # obstacle constraint
    obs_x = 1.0
    obs_y = 1.0
    obs_diam = 0.3
    for i in range(N+1):
        g.append(ca.sqrt((X[0, i]-obs_x)**2+(X[1, i]-obs_y)**2)) # should be smaller als 0.0

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
        lbx = ca.vertcat(lbx, -v_max, -omega_max)
        ubx = ca.vertcat(ubx, v_max, omega_max)
    # second, control constraint bounds
    for _ in range(N+1):
        lbx = ca.vertcat(lbx, -2.0, -2.0, -np.inf)
        ubx = ca.vertcat(ubx, 2.0, 2.0, np.inf)

    # Constraint Bounds
    lbg = []
    ubg = []

    # first, state dynamics constraint Bounds
    for _ in range(N+1):
        lbg = ca.vertcat(lbg, 0.0, 0.0, 0.0)
        ubg = ca.vertcat(ubg, 0.0, 0.0, 0.0)
    # second, obstacle constraint Bounds
    for _ in range(N+1):
        lbg = ca.vertcat(lbg, obs_diam)
        ubg = ca.vertcat(ubg, np.inf)

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
        time_list.append(solve_toc)

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

        # update initial state
        mpciter = mpciter + 1
        # break

    time_list = np.array(time_list)

    print(f"Average solve time: {time_list.mean()}")
    print(f"Minimum solve time: {time_list.min()}")

    ############################################
    # Average solve time: 0.004277663230895996 #
    # Minimum solve time: 0.002220869064331054 #
    ############################################

    draw_result = Draw_MPC_Obstacle(
        rob_diam=0.3, 
        init_state=x0.full(), 
        target_state=xs,
        robot_states=result_x,
        predict_state=predict_x, 
        obstacle=np.array([obs_x, obs_y, obs_diam/2.]), 
        export_fig=False
    )