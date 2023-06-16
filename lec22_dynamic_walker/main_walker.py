from matplotlib import pyplot as plt
import numpy as np

from scipy import interpolate
from scipy.integrate import solve_ivp

def cos(x):
    return np.cos(x)

def sin(x):
    return np.sin(x)

class Parameters:
    def __init__(self):
        self.M = 1
        self.I = 0
        self.l = 1.0
        self.g = 9.81
        self.gam = 0.1
        
        self.Kp = 5
        
        self.pause = 0.05
        self.fps = 30

def controller(z0, theta_dot_desire, params):
    theta1, theta1_dot = z0
    
    output = -params.Kp * (theta1_dot - theta_dot_desire)
    
    return output

# output이 0이면 충돌이 일어났다는 뜻
def collision(t, z, phi, M, I, l, g, gam):

    output = 1
    theta1, omega1 = z

    if (theta1 > -0.05):
        output = 1
    else:
        output = 2 * theta1 + phi

    return output

def midstance(t, z, phi, M, I, l, g, gam):
    theta, omega = z
    
    return theta

def single_stance(t, z, phi, M, I, l, g, gam):
    theta1, omega1 = z
    
    A_ss = I + M*l*l
    b_ss = -M*g*l*sin(gam - theta1)
    
    alpha1 = b_ss / A_ss
    
    return [omega1, alpha1]
    
def footstrike(t_minus, z_minus, phi, params):
    
    theta1_n, omega1_n = z_minus
    
    M = params.M
    I = params.I
    l = params.l
    g = params.g
    gam = params.gam

    theta1_plus = theta1_n + phi
    
    J_fs = np.zeros((2,3))
    A_fs = np.zeros((3,3))

    b_fs = np.zeros((5,1))
    
    J11 = 1;
    J12 = 0;
    J13 = l*(cos(phi + theta1_n) - cos(theta1_n));
    J21 = 0;
    J22 = 1;
    J23 = l*(sin(phi + theta1_n) - sin(theta1_n));
    
    J_fs = np.array([
        [J11, J12, J13], 
        [J21, J22, J23]
    ])
    
    A11 = M;
    A12 = 0;
    A13 = -M*l*cos(theta1_n);
    A21 = 0;
    A22 = M;
    A23 = -M*l*sin(theta1_n);
    A31 = -M*l*cos(theta1_n);
    A32 = -M*l*sin(theta1_n);
    A33 = I + M*l*l;

    A_fs = np.array([
        [A11, A12, A13], 
        [A21, A22, A23], 
        [A31, A32, A33]
    ])
    
    
    M_fs = np.block([
        [A_fs, -np.transpose(J_fs) ], 
        [J_fs, np.zeros((2,2))] 
    ])
    
    b_fs = np.block([
        A_fs.dot([0, 0, omega1_n]), 0, 0
    ])
    
    x_hs = np.linalg.inv(M_fs).dot(b_fs)
    
    omega1_plus = x_hs[2]
    
    return [theta1_plus, omega1_plus]
    
def one_step(step_i, z0, t0, phi, xh_start, params):
    
    z_output = []
    t_output = []

    t_start = t0
    t_end   = t_start + 4
    t = np.linspace(t_start, t_end, 1001)
    
    # first swing
    collision.terminal = True
    sol = solve_ivp(
        single_stance, [t_start, t_end], z0, method='RK45', t_eval=t,
        dense_output=True, events=collision, atol = 1e-13, rtol = 1e-12, 
        args=(phi, params.M,params.I,params.l,params.g,params.gam)
    )
    # TODO: Check ERR

    t_first_swing = sol.t
    m, n = np.shape(sol.y) # m : 3 / n : sth
    z_first_swing = np.zeros((n, m))
    z_first_swing = sol.y.T

    xh_temp1 = xh_start + params.l*sin(z_first_swing[0,0]) - params.l*sin(z_first_swing[:,0]); 
    yh_temp1 = params.l * cos(z_first_swing[:,0]);
    
    if(step_i % 2 == 0):
        xb_foot1 = xh_temp1 + params.l*sin(z_first_swing[:,0]);
        yb_foot1 = yh_temp1 - params.l*cos(z_first_swing[:,0]);
        xa_foot1 = xh_temp1 + params.l*sin(phi + z_first_swing[:,0]);
        ya_foot1 = yh_temp1 - params.l*cos(phi + z_first_swing[:,0]);
    else:
        xa_foot1 = xh_temp1 + params.l * sin(z_first_swing[:,0]);
        ya_foot1 = yh_temp1 - params.l * cos(z_first_swing[:,0]);
        xb_foot1 = xh_temp1 + params.l * sin(phi + z_first_swing[:,0]);
        yb_foot1 = yh_temp1 - params.l * cos(phi + z_first_swing[:,0]);
    
    z_temp = np.concatenate(
        (z_first_swing, 
            xh_temp1.reshape(len(xh_temp1), 1), yh_temp1.reshape(len(yh_temp1), 1),
            xa_foot1.reshape(len(xa_foot1), 1), ya_foot1.reshape(len(ya_foot1), 1),
            xb_foot1.reshape(len(xb_foot1), 1), yb_foot1.reshape(len(yb_foot1), 1),
        ), axis=1)
    
    z_output = z_temp
    t_output = t_first_swing
    
    # foot strike는 z_minus와 t_minus를 준비해서 footstrike 함수에 넣어준다.
    z_minus = np.array(sol.y_events[-1][0,:])
    t_minus = sol.t_events[-1][0]

    z_plus = footstrike(t_minus, z_minus, phi, params)
    
    z0 = z_plus
    t0 = t_minus
    
    t_end = t0 + 4
    t = np.linspace(t0, t_end, 1001)
    
    # second swing
    midstance.terminal = True
    sol = solve_ivp(
        single_stance, [t0, t_end], z0, method='RK45', t_eval=t,
        dense_output=True, events=midstance, atol = 1e-13, rtol = 1e-12, 
        args=(phi, params.M,params.I,params.l,params.g,params.gam)
    )
    # TODO: Check ERR
    
    t_second_swing = sol.t
    m, n = np.shape(sol.y) # m : 3 / n : sth
    z_second_swing = np.zeros((n, m))
    z_second_swing = sol.y.T

    xh_start = xh_temp1[-1]
    xh_temp2 = xh_start + params.l*sin(z_second_swing[0,0]) - params.l*sin(z_second_swing[:,0]); 
    yh_temp2 = params.l * cos(z_second_swing[:,0])
    
    m = len(t_second_swing)
    phi_traj = np.linspace(2*z_first_swing[-1,0], phi, m).T

    if(step_i % 2 == 0):
        xa_foot2 = xh_temp2 + params.l*sin(z_second_swing[:,0]);
        ya_foot2 = yh_temp2 - params.l*cos(z_second_swing[:,0]);
        xb_foot2 = xh_temp2 + params.l*sin(phi_traj+z_second_swing[:,0]);
        yb_foot2 = yh_temp2 - params.l*cos(phi_traj+z_second_swing[:,0]);
    else:
        yb_foot2 = yh_temp2 - params.l*cos(z_second_swing[:,0]);
        xb_foot2 = xh_temp2 + params.l*sin(z_second_swing[:,0]);
        xa_foot2 = xh_temp2 + params.l*sin(phi_traj+z_second_swing[:,0]);
        ya_foot2 = yh_temp2 - params.l*cos(phi_traj+z_second_swing[:,0]);        

    z_temp = np.concatenate((
        z_second_swing,
        xh_temp2.reshape(len(xh_temp2), 1), yh_temp2.reshape(len(yh_temp2), 1),
        xa_foot2.reshape(len(xa_foot2), 1), ya_foot2.reshape(len(ya_foot2), 1),
        xb_foot2.reshape(len(xb_foot2), 1), yb_foot2.reshape(len(yb_foot2), 1)
    ), axis=1)
    
    z_output = np.concatenate((z_output, z_temp), axis=0)
    t_output = np.concatenate((t_output, t_second_swing), axis=0)
    
    return z_output, t_output

def n_steps(z0, t0, step_size, theta_dot_desire, params):
    
    # xh_start, yh_start : hip position
    xh_start, yh_start = 0, params.l * cos(z0[0])

    t = np.array([t0])

    # theta2 (=phi)에 대한 정보가 없기 때문에 leg endpoint도 싹다 집어넣는다.
    # theta1 theta1_dot xh yh x_c1 y_c1 x_c2 y_c2
    z = np.zeros((1, 8))
    x_c1, y_c1 = xh_start + params.l * sin(z0[0]), yh_start - params.l * cos(z0[0])
    x_c2, y_c2 = xh_start + params.l * sin(np.pi/6 + z0[0]), yh_start - params.l * cos(np.pi/6 + z0[0])
    z[0] = np.append(z0, np.array([
        xh_start, yh_start,
        x_c1, y_c1,
        x_c2, y_c2
    ]))

    for i in range(step_size):
        # phi = controller(z0, theta_dot_desire[i], params)
        phi = np.pi/6
        z_temp, t_temp = one_step(i, z0, t0, phi, xh_start, params)
        
        z = np.concatenate((z, z_temp), axis=0)
        t = np.concatenate((t, t_temp), axis=0)

        theta1, omega1 = z_temp[-1,0], z_temp[-1,1]
        z0 = np.array([theta1, omega1])
        t0 = t_temp[-1]

        # one step에서 zz_temp[-1] 스위칭이 일어나기 때문에 [-2] 사용
        # xh_start = zz_temp[-2,4]
        xh_start = z_temp[-2,2]
        
    return z, t

def animate(t, z, parms):
    #interpolation
    data_pts = 1/parms.fps
    t_interp = np.arange(t[0],t[len(t)-1],data_pts)

    [m, n] = np.shape(z)
    z_interp = np.zeros((len(t_interp),n))

    for i in range(0,n):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    l = parms.l
    min_xh = min(z[:,4]) 
    max_xh = max(z[:,4])

    dist_travelled = max_xh - min_xh;
    camera_rate = dist_travelled/len(t_interp);

    window_xmin = -1*l; window_xmax = 1*l;
    window_ymin = -0.1; window_ymax = 1.1*l;

    R1 = np.array([min_xh-l,0])
    R2 = np.array([max_xh+l,0])

    # 바닥은 처음에 다 그려버린다.
    ramp, = plt.plot([R1[0], R2[0]],[R1[1], R2[1]],linewidth=5, color='black')
    
    for i in range(0,len(t_interp)):
        theta1 = z_interp[i,0]
        xh, yh = z_interp[i,2], z_interp[i,3]
        xfa, yfa = z_interp[i,4], z_interp[i,5]
        xfb, yfb = z_interp[i,6], z_interp[i,7]
        
        leg1, = plt.plot([xh, xfa],[yh, yfa],linewidth=5, color='blue')
        leg2, = plt.plot([xh, xfb],[yh, yfb],linewidth=5, color='red')
        hip, = plt.plot(xh, yh, color='black', marker='o', markersize=10)

        window_xmin = window_xmin + camera_rate;
        window_xmax = window_xmax + camera_rate;
        plt.xlim(window_xmin,window_xmax)
        plt.ylim(window_ymin,window_ymax)
        plt.gca().set_aspect('equal')
        
        plt.pause(parms.pause)
        hip.remove()
        leg1.remove()
        leg2.remove()
        
    plt.close()

if __name__=="__main__":
    
    params = Parameters()

    # theta_dot_des = [-0.5, -0.5]
    theta_dot_des = [-0.5, -1, -1.2, -0.9, -0.7, -0.7, -1, -1.5]
    steps = len(theta_dot_des)
    
    t0 = 0.0
    z0 = [0, theta_dot_des[0]]
    
    z, t = n_steps(z0, t0, steps, theta_dot_des, params)
    animate(t, z, params)

