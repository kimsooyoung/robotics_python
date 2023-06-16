import numpy as np

import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

class Params():
    def __init__(self):
        self.g = 10.0
        self.ground = 0.0
        self.l = 1
        self.m = 1
        self.k = 500
        
        self.Kp = 0.15
        
        # new params for raibert hopper
        self.control_tp = np.pi * np.sqrt(self.m/self.k)
        self.control_theta = 0.0
        
        self.theta = 10 * (np.pi / 180)
        self.T = np.pi * np.sqrt(self.m/self.k)
        
        # self.vdes = [0.0, 0.0]
        # self.vdes = [0.0, 0.3, 0.1, 0.0, 0.0, 0.0]
        self.vdes = [0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3]

        self.pause = 0.01
        self.fps = 10

def flight(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    
    return [x_dot, 0, y_dot, -g]

def contact(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    # contact event
    return y - l0 * np.cos(theta)

def stance(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    
    l = np.sqrt(x**2 + y**2)
    
    F_spring = k * (l0 - l)
    Fx_spring = F_spring*(x/l)
    Fy_spring = F_spring*(y/l)
    Fy_gravity = m*g
    xddot = (1/m)*(Fx_spring)
    yddot = (1/m)*(-Fy_gravity+Fy_spring)
    
    return [x_dot, xddot, y_dot, yddot]

def release(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    
    l = np.sqrt(x**2 + y**2)
    # release event
    return l - l0

def apex(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    # apex event
    return y_dot - 0

def raibert_controller(vx, vx_des, params):
    
    theta_c = vx * params.control_tp/(2*params.l)
    speed_correction = params.Kp*(vx - vx_des)
    theta = np.arcsin(theta_c)+speed_correction
    
    params.control_theta = theta
    
def one_step(z0, t0, params, i):
    
    dt = 1.0
    
    vx_apex, vx_des = z0[1], params.vdes[i]
    raibert_controller(vx_apex, vx_des, params)
    m, g, l0, k, theta = params.m, params.g, params.l, params.k, params.control_theta
    ground = params.ground
    
    print("vx_apex: {}, vx_des: {}, theta: {}".format(vx_apex, vx_des, theta))
    
    t_span = np.linspace(t0, t0+dt, 1001)
    
    contact.direction = -1
    contact.terminal = True
    
    contact_sol = solve_ivp(
        flight, [t0, t0+dt], z0, method='RK45', t_eval=t_span,
        dense_output=True, events=contact, atol = 1e-13, rtol = 1e-13, 
        args=(m, g, l0, k, theta)
    )
    
    t_contact = contact_sol.t
    m, n = contact_sol.y.shape
    z_contact = contact_sol.y.T
    
    fp_x = z_contact[:,0] + l0*np.sin(theta)
    fp_y = z_contact[:,2] - l0*np.cos(theta)
    
    t0 = t_contact[-1]
    z0 = z_contact[-1]
    
    z_contact = np.concatenate((
        z_contact, 
        fp_x.reshape(-1,1), 
        fp_y.reshape(-1,1)
    ), axis=1)
    
    x_com = z0[0]
    z0[0] = -l0 * np.sin(theta)
    
    x_foot = x_com + l0*np.sin(theta)
    y_foot = ground
    
    ### stance phase ###
    
    t_span = np.linspace(t0, t0+dt, 1001)
    
    release.direction = 1
    release.terminal = True
    
    release_sol = solve_ivp(
        stance, [t0, t0+dt], z0, t_eval=t_span,
        dense_output=True, events=release, atol = 1e-13, rtol = 1e-13, 
        args=(m, g, l0, k, theta)
    )
    
    t_release = release_sol.t
    m, n = release_sol.y.shape
    z_release = release_sol.y.T
    z_release[:,0] = z_release[:,0] + x_com + l0*np.sin(theta)
    
    z0 = z_release[-1]
    t0 = t_release[-1]
    
    fp_x = x_foot * np.ones((n,1))
    fp_y = y_foot * np.zeros((n,1))
    
    z_release = np.concatenate((
        z_release, 
        fp_x.reshape(-1,1), 
        fp_y.reshape(-1,1)
    ), axis=1)
    
    ### apex stance ###
    t_span = np.linspace(t0, t0+dt, 1001)
    
    apex.direction = 0
    apex.terminal = True
    
    apex_sol = solve_ivp(
        flight, [t0, t0+dt], z0, method='RK45', t_eval=np.linspace(t0, t0+dt, 1001),
        dense_output=True, events=apex, atol = 1e-13, rtol = 1e-13, 
        args=(m, g, l0, k, theta)
    )
    
    t_apex = apex_sol.t
    m, n = apex_sol.y.shape
    z_apex = apex_sol.y.T
    
    fp_x = z_apex[:,0] + l0*np.sin(theta)
    fp_y = z_apex[:,2] - l0*np.cos(theta)
    
    t0 = t_apex[-1]
    z0 = z_apex[-1]
    
    z_apex = np.concatenate((
        z_apex, 
        fp_x.reshape(-1,1), 
        fp_y.reshape(-1,1)
    ), axis=1)
    
    t_total = np.concatenate((t_contact, t_release, t_apex), axis=0)
    z_total = np.concatenate((z_contact, z_release, z_apex), axis=0)
    
    return t_total, z_total
    
def n_step(z0, params, step):
    
    x0, x0dot, y0, y0dot = z0
    
    t0 = 0
    
    z = np.zeros((1,6))
    t = np.zeros(1)
    
    t[0] = 0
    z[0] = [
        *z0,
        x0 + params.l*np.sin(params.control_theta),
        y0 - params.l*np.cos(params.control_theta)
    ]
    
    for i in range(step):
        t_total, z_total = one_step(z0, t0, params, i)
        
        t0 = t_total[-1]
        z0 = z_total[-1,:-2]
        
        
        t = np.concatenate((t, t_total), axis=0)
        z = np.concatenate((z, z_total), axis=0)
        
    return t, z


def animate(z, t, parms):
    #interpolation
    data_pts = 1/parms.fps
    t_interp = np.arange(t[0], t[len(t)-1], data_pts)
    m, n = np.shape(z)
    shape = (len(t_interp),n)
    z_interp = np.zeros(shape)

    for i in range(0, n):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    l = parms.l

    min_xh = min(z[:,0]); max_xh = max(z[:,0]);
    # print(f"min_xh: {min_xh}, max_xh: {max_xh}")
    dist_travelled = max_xh - min_xh;
    camera_rate = dist_travelled/len(t_interp);

    window_xmin = -3.0*l; window_xmax = 3.0*l;
    window_ymin = -0.1; window_ymax = 3.0*l;

    #plot
    for i in range(0,len(t_interp)):

        x, y = z_interp[i,0], z_interp[i,2]
        x_foot, y_foot = z_interp[i,4], z_interp[i,5]

        leg, = plt.plot([x, x_foot],[y, y_foot],linewidth=2, color='black')
        hip, = plt.plot(x, y, color='red', marker='o', markersize=10)

        window_xmin = window_xmin + camera_rate;
        window_xmax = window_xmax + camera_rate;
        plt.xlim(window_xmin,window_xmax)
        plt.ylim(window_ymin,window_ymax)
        plt.gca().set_aspect('equal')

        plt.pause(parms.pause)
        hip.remove()
        leg.remove()

    plt.close()
 
if __name__=="__main__":
    
    params = Params()
    
    step = len(params.vdes)
    
    x0dot = params.vdes[0]
    y0 = 1.2
    
    z0 = [0, x0dot, y0, 0]
    
    t, z = n_step(z0, params, step)
    animate(z, t, params)