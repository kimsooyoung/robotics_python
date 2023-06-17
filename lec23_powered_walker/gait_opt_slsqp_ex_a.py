import numpy as np
from matplotlib import pyplot as plt

import scipy.optimize as opt
from scipy.integrate import solve_ivp, odeint

from main_walker import single_stance, single_stance2, single_stance_ode_int
from main_walker import footstrike, animate, plot

from main_walker import Parameters as WalkerParameters

# control P = 0.1
# x0에 torque 항 u_opt 추가
# u_opt upper/lower bound 설정
# simulator에서 u_opt 적용한 single_stance2 for loop로 실행 
# cost에서 u_opt 제곱 sum 추가

class OptParams:
    def __init__(self):
        self.N = 4

def cost(x, args):
    return 0
    # return x[0]
    # N = OptParams().N;
    
    # time = x[0]
    # dt = time / N
    # u_opt = x[9:9+N+1]

    # # add torque^2 dt
    # F = sum([ x*y for x,y in zip(u_opt, u_opt)] )
    # return time

def simulator(x):
    
    z_ss0, z_bfs = x[1:5], x[5:9]
    u_opt = x[9:9+N+1]
    
    params = WalkerParameters()
    params.P = 0.1
    
    t_start = 0
    t_end = x[0]
    t_span = np.linspace(t_start, t_end, 100)
    t_opt = np.linspace(t_start, t_end, N+1)
    
    # single_stance2에서 문제가 있는 상황이다.
    sol = solve_ivp(
        single_stance2, [t_start, t_end], z_ss0, method='RK45', t_eval=t_span,
        dense_output=True, atol = 1e-13, rtol = 1e-13, 
        args=(params.M,params.m,params.I,params.l,params.c,params.g,params.gam, t_opt, u_opt)
    )
    
    # sol = solve_ivp(
    #     single_stance, [t_start, t_end], z_ss0, method='RK45', t_eval=t_span,
    #     dense_output=True, atol = 1e-13, rtol = 1e-13, 
    #     args=(params.M,params.m,params.I,params.l,params.c,params.g,params.gam)
    # )
    
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

def simulator2(x):

    N = OptParams().N

    time = x[0]
    z_ss0 = x[1:5]
    z_bfs = x[5:9]
    u_opt = x[9:9+N+1]
    # print(x, u_opt)
    
    params = WalkerParameters()
    params.P = 0.1
    
    t_opt = np.linspace(0, time, N+1)
    z = np.zeros((N+1, 4))
    z[0] = z_ss0
    
    dynamics_args = (params.M,params.m,params.I,params.l,params.c,params.g,params.gam)
    
    for i in range(0, N):
        args = dynamics_args + (t_opt[i],t_opt[i+1],u_opt[i], u_opt[i+1])
        z_temp = odeint(
            single_stance_ode_int, z[i], np.array([t_opt[i], t_opt[i+1]]), 
            args, atol = 1e-13, rtol = 1e-13
        )
        z[i+1] = z_temp[-1]
    
    # z0 = z_ss0
    
    # z = np.zeros((1, 4))
    # t = np.zeros(1)
    # z[0] = z0
    # t[0] = 0
    
    # for i in range(N):
    #     t_span = np.linspace(t_opt[i], t_opt[i+1], 2)
    #     sol = solve_ivp(
    #         single_stance2, [t_opt[i], t_opt[i+1]], z0, method='RK45', t_eval=t_span,
    #         dense_output=True, atol = 1e-13, rtol = 1e-13, 
    #         args=(params.M,params.m,params.I,params.l,params.c,params.g,params.gam, t_opt, u_opt)
    #     )
        
    #     t_temp = sol.t
    #     m, n = np.shape(sol.y)
    #     z_temp = np.zeros((n, m))
    #     z_temp = sol.y.T
        
    #     z0 = z_temp[-1]
    #     z = np.concatenate((z, np.expand_dims(np.array(z0), axis=0)), axis=0)        
    #     t = np.concatenate((t, np.expand_dims(np.array(t_temp[-1]), axis=0)), axis=0)    
    
    z_ssT = z[-1]
    z_afs = footstrike( 0, z_bfs, params);
    
    l = params.l
    xh = l * np.sin(z[0,0]) - l * np.sin(z[:,0])
    yh = l * np.cos(z[:,0])
    
    # Convert the list to a 2D array
    xh = np.expand_dims(np.array(xh), axis=1)
    yh = np.expand_dims(np.array(yh), axis=1)
    z_output = np.concatenate((z, xh, yh), axis=1)

    return z_ssT, z_afs, t_opt, z_output

def walker_constraint(x):
    
    time = x[0]
    z_ss0 = x[1:5]
    z_bfs = x[5:9]

    # theta1, omega1, theta2, omega2
    theta1_bfs = z_bfs[0]
    theta2_bfs = z_bfs[2]
    collision_condition = theta2_bfs + 2*theta1_bfs
    
    z_ssT, z_afs, _, _ = simulator(x)
    # z_ssT, z_afs, _, _ = simulator2(x)

    swing_state_diff = z_ss0 - z_afs
    strike_state_diff = z_bfs - z_ssT
    
    # debugging
    print(f"swing_state_diff: {swing_state_diff}")
    print(f"strike_state_diff: {strike_state_diff}")
    opt_funcs = [ *swing_state_diff, *strike_state_diff, collision_condition ]
    
    return opt_funcs

# Define a callback function to display step results
def callback(xk):
    print('Step result:', xk)

def torque_plot(opt_result):
    
    plt.figure(1)

    plt.plot(opt_result,'r--')
    plt.ylabel('torque')

    plt.show(block=False)
    plt.pause(3)
    plt.close()

if __name__=="__main__":
    
    opt_params = OptParams()
    
    # control sampling
    N = opt_params.N
    
    time_min, time_max = 1, 3
    u_min, u_max = -0.5, 0.5 
    
    # theta1, omega1, theta2, omega2
    z_ss_lb = [0.1, -0.5, -2*0.3, -0.5]; z_ss_ub = [0.3, -0.1, -2*0.1, 0.5]
    z_bfs_lb = [-0.3, -0.5, 2*0.1, -0.5]; z_bfs_ub = [-0.1, -0.1, 2*0.3, 0.5]

    #####################################################
    ######## example a. powered walker optimize #########
    #####################################################
    
    t_bf_strike = 3
    z_ini = [0.15, -0.2, -0.3, 0]
    z_bf_strike = [-0.15, -0.2, 0.3, 0]
    u_opt = (u_min + (u_max-u_min) * np.random.rand(1, N+1)).flatten()
    print(u_opt)
    
    # example a. powered walker optimize
    u_lb = (u_min * np.ones((1,N+1))).flatten()
    u_ub = (u_max * np.ones((1,N+1))).flatten()
    x0 = [ t_bf_strike, *z_ini, *z_bf_strike, *u_opt ]
    x_min = [ time_min, *z_ss_lb, *z_bfs_lb, *u_lb ]
    x_max = [ time_max, *z_ss_ub, *z_bfs_ub, *u_ub ]
    
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
    
    print(f"total torque : {opt_state[8:]}")
    
    for i in range(9):
        print(opt_state[i])
    
    print('Copy paste in main_walker.py')
    print(f"theta1, omega1, theta2, omega2 = {opt_state[1]}, {opt_state[2]}, {opt_state[3]}, {opt_state[4]}")
    
    z_ssT, z_afs, t, z_output = simulator(x0)
    # z_ssT, z_afs, t, z_output = simulator2(x0)
    walker_param = WalkerParameters()
    # animate(t, z_output, walker_param)
    # torque_plot(opt_state)