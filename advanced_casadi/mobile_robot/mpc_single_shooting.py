import casadi as ca
import casadi.tools as ca_tools

import time
import numpy as np
from draw import Draw_MPC_point_stabilization_v1

def shift_movement(T, t0, x0, u, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T*f_value
    t = t0 + T
    u_end = ca.horzcat(u[:, 1:], u[:, -1])

    return t, st, u_end.T

if __name__ == '__main__':

    T = 0.2 # sampling time [s]
    N = 100 # prediction horizon
    
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
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC

    # N Control Variables 
    U = ca.SX.sym('U', n_controls, N)
    # N + 1 State variables
    X = ca.SX.sym('X', n_states, N+1)
    # Initial and Final states
    P = ca.SX.sym('P', n_states + n_states)

    # initial condition
    X[:, 0] = P[:3]

    # define the relationship within the horizon

    # compute solution symbolically - Euler integration
    for i in range(N):
        f_value = f(X[:, i], U[:, i])
        X[:, i+1] = X[:, i] + f_value * T

    # this function to get the optimal trajectory knowing the optimal solution
    ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

    # weighing matrices (states)
    Q = np.diag([1, 5, 0.1])
    # weighing matrices (controls)
    R = np.diag([0.5, 0.05])

    # Objective function
    obj = 0

    # compute objective
    for i in range(N):
        obj = obj + (X[:3, i] - P[3:6]).T @ Q @ (X[:3, i] - P[3:6]) + U[:, i].T @ R @ U[:, i]

    # compute constraints
    g = []
    for i in range(N+1):
        g.append(X[0, i])
        g.append(X[1, i])

    nlp_prob = {
        'f': obj, 
        'x': ca.reshape(U, -1, 1), 
        'g': ca.vertcat(*g),
        'p': P
    }
    opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    # THE SIMULATION LOOP SHOULD START FROM HERE
    lbx = []
    ubx = []
    lbg = -2.0
    ubg = 2.0

    for _ in range(N):
        lbx = ca.vertcat(lbx, -v_max, -omega_max)
        ubx = ca.vertcat(ubx, v_max, omega_max)

    
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1) # initial state
    xs = np.array([1.5, 1.5, 0.0]).reshape(-1, 1) # final state
    u0 = np.array([0.0, 0.0]*N).reshape(-1, 2)    # controls

    #####################

    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [] # for the time
    xx = []
    sim_time = 20.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    c_p = np.concatenate((x0, xs))
    init_control = ca.reshape(u0, -1, 1)
    res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    lam_x_ = res['lam_x']
    ### inital test
    while(np.linalg.norm(x0-xs)>1e-2 and mpciter-sim_time/T<0.0 ):
        ## set parameter
        c_p = np.concatenate((x0, xs))
        init_control = ca.reshape(u0, -1, 1)
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx, lam_x0=lam_x_)
        lam_x_ = res['lam_x']
        # res = solver(x0=init_control, p=c_p,)
        # print(res['g'])
        index_t.append(time.time()- t_)
        u_sol = ca.reshape(res['x'], n_controls, N) # one can only have this shape of the output
        ff_value = ff(u_sol, c_p) # [n_states, N+1]
        x_c.append(ff_value)
        u_c.append(u_sol[:, 0])
        t_c.append(t0)
        t0, x0, u0 = shift_movement(T, t0, x0, u_sol, f)

        x0 = ca.reshape(x0, -1, 1)
        xx.append(x0.full())
        mpciter = mpciter + 1
    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time)/(mpciter))
    draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0.full(), target_state=xs, robot_states=xx, export_fig=False)