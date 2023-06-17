import numpy as np
from matplotlib import pyplot as plt

import scipy.optimize as opt
from scipy.integrate import solve_ivp

from main_walker import single_stance, footstrike, animate, plot
from main_walker import Parameters as WalkerParameters

def cost(x, args):
    return x[0]

def simulator(x):
    
    z_ss0, z_bfs = x[1:5], x[5:9]
    params = WalkerParameters()
    
    t_start = 0 
    t_end = x[0]
    t = np.linspace(t_start, t_end, 100)
    
    sol = solve_ivp(
        single_stance, [t_start, t_end], z_ss0, method='RK45', t_eval=t,
        dense_output=True, atol = 1e-13, rtol = 1e-13, 
        args=(params.M,params.m,params.I,params.l,params.c,params.g,params.gam)
    )
    
    t = sol.t
    # m : 4 / n : 100
    m, n = np.shape(sol.y)
    z = np.zeros((n, m))
    z = sol.y.T
    
    z_ssT = z[-1]
    z_afs = footstrike( 0, z_bfs, params);
    
    l = params.l
    xh = l * np.sin(z[0,0]) - l * np.sin(z[:,0])
    yh = l * np.cos(z[:,0])
    
    # Convert the list to a 2D array
    xh = np.expand_dims(np.array(xh), axis=1)
    yh = np.expand_dims(np.array(yh), axis=1)
    z_output = np.concatenate((z, xh, yh), axis=1)

    return z_ssT, z_afs, t, z_output

def walker_constraint(x):
    
    time = x[0]
    z_ss0 = x[1:5]
    z_bfs = x[5:9]
    
    # theta1, omega1, theta2, omega2
    theta1_bfs = z_bfs[0]
    theta2_bfs = z_bfs[2]
    collision_condition = theta2_bfs + 2*theta1_bfs
    
    z_ssT, z_afs, _, _ = simulator(x)
        
    swing_state_diff = z_ss0 - z_afs
    strike_state_diff = z_bfs - z_ssT
    
    # debugging
    print(f"swing_state_diff: {swing_state_diff}")
    print(f"strike_state_diff: {strike_state_diff}")
    opt_funcs = [ *swing_state_diff, *strike_state_diff, collision_condition ]
    
    return opt_funcs 

if __name__=="__main__":
    
    # theta1, omega1, theta2, omega2
    #####################################################
    ######## example1. passive walker optimize ##########
    #####################################################
    
    # optimal case
    # t_bf_strike = 2.4495703707513576
    # z_ini = [0.18350082, -0.27333599, -0.36700164, 0.03138302]
    # z_bf_strike = [-0.18307195, -0.27285737, 0.36695179, 0.03209671]
    
    # random guess
    # t_bf_strike = np.random.uniform(1, 3)
    t_bf_strike = 3
    z_ini = [0.15, -0.2, -0.3, 0]
    z_bf_strike = [-0.15, -0.2, 0.3, 0]
    
    x0 = [ t_bf_strike, *z_ini, *z_bf_strike ]
    
    time_min, time_max = 1, 3
    z_ss_lb = [0.1, -0.5, -2*0.3, -0.5]; z_ss_ub = [0.3, -0.1, -2*0.1, 0.5]
    z_bfs_lb = [-0.3, -0.5, 2*0.1, -0.5]; z_bfs_ub = [-0.1, -0.1, 2*0.3, 0.5]
    
    x_min = [ time_min, *z_ss_lb, *z_bfs_lb ]
    x_max = [ time_max, *z_ss_ub, *z_bfs_ub ]  
    
    limits = opt.Bounds(x_min, x_max)
    
    constraint = {
        'type': 'eq',
        'fun': walker_constraint
    }
    
    result = opt.minimize(
        cost, x0, args=(WalkerParameters), method='SLSQP', 
        constraints=[constraint], 
        options={'ftol': 1e-6, 'disp': True, 'maxiter':500},
        bounds=limits
    )
    opt_state = result.x
    print_result = [ print(i) for i in opt_state ]
    
    print('Copy paste in main_walker.py')
    print(f"theta1, omega1, theta2, omega2 = {opt_state[1]}, {opt_state[2]}, {opt_state[3]}, {opt_state[4]}")
    
    z_ssT, z_afs, t, z_output = simulator(x0)
    walker_param = WalkerParameters()
    animate(t, z_output, walker_param)
    plot(t, z_output)