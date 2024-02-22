#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools

import numpy as np
from draw import Draw_MPC_point_stabilization_v1, draw_gt, draw_gt_measurements, draw_gtmeas_noisemeas, draw_gt_mhe_measurements


def shift_movement(T, t0, x0, u, x_f, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T*f_value.full()
    t = t0 + T
    u_end = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)
    x_f = np.concatenate((x_f[:, 1:], x_f[:, -1:]), axis=1)

    return t, st, u_end, x_f

if __name__ == '__main__':
    T = 0.2 # sampling time [s]
    N = 100 # prediction horizon
    rob_diam = 0.3 # [m]
    v_max = 0.6
    omega_max = np.pi/4.0

    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    states = ca.vertcat(x, y)
    states = ca.vertcat(states, theta)
    n_states = states.size()[0]

    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    n_controls = controls.size()[0]

    ## rhs
    rhs = ca.vertcat(v*ca.cos(theta), v*ca.sin(theta))
    rhs = ca.vertcat(rhs, omega)

    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    U = ca.SX.sym('U', n_controls, N)

    X = ca.SX.sym('X', n_states, N+1)

    P = ca.SX.sym('P', n_states+n_states)


    ### define
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 5.0, 0.0],[0.0, 0.0, .1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0 #### cost
    g = [] # equal constrains
    g.append(X[:, 0]-P[:3])
    for i in range(N):
        obj = obj + ca.mtimes([(X[:, i]-P[3:]).T, Q, X[:, i]-P[3:]]) + ca.mtimes([U[:, i].T, R, U[:, i]])
        x_next_ = f(X[:, i], U[:, i])*T +X[:, i]
        g.append(X[:, i+1]-x_next_)

    opt_variables = ca.vertcat( ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

    nlp_prob = {'f': obj, 'x': opt_variables, 'p':P, 'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    lbg = 0.0
    ubg = 0.0
    lbx = []
    ubx = []
    for _ in range(N):
        lbx.append(-v_max)
        lbx.append(-omega_max)
        ubx.append(v_max)
        ubx.append(omega_max)
    for _ in range(N+1): # note that this is different with the method using structure
        lbx.append(-2.0)
        lbx.append(-2.0)
        lbx.append(-np.inf)
        ubx.append(2.0)
        ubx.append(2.0)
        ubx.append(np.inf)

    # Simulation
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)# initial state
    x0_ = x0.copy()
    x_m = np.zeros((n_states, N+1))
    next_states = x_m.copy()
    xs = np.array([1.5, 1.5, 0.0]).reshape(-1, 1) # final state
    u0 = np.array([1,2]*N).reshape(-1, 2).T# np.ones((N, 2)) # controls
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = []
    # t_c = [t0] # for the time
    xx = []
    sim_time = 20.0

    ## start MPC
    mpciter = 0
    ### inital test
    while(np.linalg.norm(x0-xs)>1e-2 and mpciter-sim_time/T<0.0 ):
        ## set parameter
        c_p = np.concatenate((x0, xs))
        init_control = np.concatenate((u0.T.reshape(-1, 1), next_states.T.reshape(-1, 1)))
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        estimated_opt = res['x'].full() # the feedback is in the series [u0, x0, u1, x1, ...]
        u0 = estimated_opt[:200].reshape(N, n_controls).T # (n_controls, N)
        x_m = estimated_opt[200:].reshape(N+1, n_states).T# [n_states, N]
        x_c.append(x_m.T)
        u_c.append(u0[:, 0])
        t_c.append(t0)
        t0, x0, u0, next_states = shift_movement(T, t0, x0, u0, x_m, f)
        x0 = ca.reshape(x0, -1, 1)
        x0 = x0.full()
        xx.append(x0)
        mpciter = mpciter + 1

    # draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0_, target_state=xs, robot_states=xx )

    # convert list data to numpy array 
    xx_np = np.array(xx)
    # draw_gt(t_c, xx_np)
    
    # synthesize the measurements
    con_cov = np.diag([0.005, np.deg2rad(2)])**2
    meas_cov = np.diag([0.1, np.deg2rad(2)])**2

    r = []
    alpha = []
    for i in range(len(xx)):
        r.append(np.sqrt(xx_np[i, 0]**2+xx_np[i, 1]**2) + np.sqrt(meas_cov[0, 0])*np.random.randn())
        alpha.append(np.arctan(xx_np[i, 1]/xx_np[i, 0]) + np.sqrt(meas_cov[1, 1])*np.random.randn())
    y_measurements = np.concatenate((np.array(r).reshape(-1, 1), np.array(alpha).reshape(-1, 1)), axis=1)
    # draw_gt_measurements(t_c, xx_np, y_measurements)
    # draw_gtmeas_noisemeas(t_c, xx_np, y_measurements)
    
    ## MHE 
    u_c_np = np.array(u_c)
    T_mhe = 0.2
    N_MHE = 6 # estimation horizon
    print("MHE horizon {}".format(N_MHE))
    ### using the same model and states （states, f, next_controls)
    mhe_U = ca.SX.sym('mhe_U', n_controls, N_MHE)
    mhe_X = ca.SX.sym('mhe_X', n_states, N_MHE+1)
    n_meas = 2
    Mes_ref = ca.SX.sym('Mes_ref', n_meas, N_MHE+1)
    U_ref = ca.SX.sym('U_ref', n_controls, N_MHE)
    
    ## measurement model
    f_m = lambda x, y: ca.vertcat(*[np.sqrt(x**2+y**2), np.arctan(y/x)])
    V_mat = np.linalg.inv(np.sqrt(meas_cov))
    W_mat = np.linalg.inv(np.sqrt(con_cov))
    #### constrains
    g = [] # equal constrains
    #### multiple shooting constraints
    for i in range(N_MHE):
        x_next_ = f(mhe_X[:, i], mhe_U[:, i])*T + mhe_X[:, i]
        g.append(mhe_X[:, i+1] - x_next_)
    #### cost function
    obj_mhe = 0 #### cost
    for i in range(N_MHE+1):
        h_x = f_m(mhe_X[0, i], mhe_X[1, i])
        temp_diff_ = Mes_ref[:, i] - h_x
        obj_mhe = obj_mhe + ca.mtimes([temp_diff_.T, V_mat, temp_diff_])
    for i in range(N_MHE):
        temp_diff_ = U_ref[:, i] - mhe_U[:, i]
        obj_mhe = obj_mhe + ca.mtimes([temp_diff_.T, W_mat, temp_diff_])

    mhe_target = ca.vertcat(ca.reshape(mhe_U, -1, 1), ca.reshape(mhe_X, -1, 1))
    mhe_params = ca.vertcat(ca.reshape(U_ref, -1, 1), ca.reshape(Mes_ref, -1, 1))
    ### define MHE nlp problem, the constraints stay the same as MPC
    nlp_prob_mhe = {'f': obj_mhe, 'x': mhe_target, 'p':mhe_params, 'g':ca.vertcat(*g)}
    mhe_opts_setting = {'ipopt.max_iter':2000, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    mhe_solver = ca.nlpsol('solver', 'ipopt', nlp_prob_mhe, mhe_opts_setting)
    
    mhe_lbg = 0.0
    mhe_ubg = 0.0
    mhe_lbx = []
    mhe_ubx = []
    for _ in range(N_MHE):
        mhe_lbx.append(-v_max)
        mhe_lbx.append(-omega_max)
        mhe_ubx.append(v_max)
        mhe_ubx.append(omega_max)
    for _ in range(N_MHE+1): # note that this is different with the method using structure
        mhe_lbx.append(-2.0)
        mhe_lbx.append(-2.0)
        mhe_lbx.append(-np.inf)
        mhe_ubx.append(2.0)
        mhe_ubx.append(2.0)
        mhe_ubx.append(np.inf)
    
    ## MHE simulation
    print(f"u_c: {u_c[0]}")
    X0 = np.zeros((N_MHE+1, 3))
    U0 = np.array(u_c[:N_MHE])
    for i in range(N_MHE+1):
        X0[i] = np.array([y_measurements[i, 0]*np.cos(y_measurements[i, 1]),
        y_measurements[i, 0]*np.sin(y_measurements[i, 1]), 0.0])
    print(f"X0: {X0.shape}")
    print(f"U0: {U0.shape}")

    U_estimate = None
    X_estimate = None
    for i in range(y_measurements.shape[0]-N_MHE):
        mheiter = i
        init_mhe_control = np.concatenate((U0.reshape(-1, 1), X0.reshape(-1, 1)))
        
        mhe_c_p = np.concatenate((np.array(u_c[i:i+N_MHE]).reshape(-1, 1), y_measurements[i:i+N_MHE+1].reshape(-1, 1)))
        
        print(f"init_mhe_control.shape: {init_mhe_control.shape}, mhe_c_p.shape: {mhe_c_p.shape}")
        mhe_res = mhe_solver(x0=init_mhe_control, p=mhe_c_p, lbg=mhe_lbg, lbx=mhe_lbx, ubg=mhe_ubg, ubx=mhe_ubx)
        mhe_estimated = mhe_res['x'].full()
        
        u_sol = mhe_estimated[:n_controls*N_MHE].reshape(N_MHE, n_controls) # (N_MHE, -1)
        state_sol = mhe_estimated[n_controls*N_MHE:].reshape(N_MHE+1, n_states)#
        if U_estimate is None:
            U_estimate = u_sol[N_MHE-1:]
        else:
            U_estimate = np.concatenate((U_estimate, u_sol[N_MHE-1:]))
        if X_estimate is None:
            X_estimate = state_sol[N_MHE:]
        else:
            X_estimate = np.concatenate((X_estimate, state_sol[N_MHE:]))
        # shift trajectories to initialize the next step
        X0 = np.concatenate((state_sol[1:], state_sol[-1:]))
        U0 = np.concatenate((u_sol[1:], u_sol[-1:]))
    # draw_gt_mhe_measurements(t_c, xx_np, y_measurements, X_estimate, n_mhe=N_MHE)