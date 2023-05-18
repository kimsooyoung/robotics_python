import numpy as np

from scipy import interpolate
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

class Params():
    def __init__(self):
        self.g = 10.0
        self.ground = 0.0
        self.l = 1
        self.m = 1
        self.k = 100
        self.pause = 0.1
        self.fps = 10
        
        self.Kp = 0.05
        
        # new params for raibert hopper
        self.control_tp = np.pi * np.sqrt(self.m/self.k)
        self.control_theta = 0.0
        
        self.theta = 10 * (np.pi / 180)
        self.T = np.pi * np.sqrt(self.m/self.k)
        
        # TODO: Adjust Kp 
        self.vdes = [0.0, 0.2]

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

def one_step(z0, t0, params, i):
    
    dt = 1.0
    
    vx_apex = z0[1]
    # v_act = 
    theta_c = vx_apex*params.control_tp/(2*params.l)
    speed_correction = params.Kp*(vx_apex-params.vdes[i])
    # TODO: check arcsin
    params.control_theta = np.arcsin(theta_c)+speed_correction
    m, g, l0, k, theta = params.m, params.g, params.l, params.k, params.control_theta
    
    contact_sol = solve_ivp(
        flight, [t0, t0+dt], z0, method='RK45', t_eval=np.linspace(t0, t0+dt, 1001),
        dense_output=True, events=contact, atol = 1e-13, rtol = 1e-13, 
        args=(m, g, l0, k, theta)
    )
    
    
    
    
def n_step(z0, params, step):
    
    x0, x0dot, y0, y0dot = z0
    
    t0 = 0
    dt = 1
    
    z = np.zeros((1,6))
    t = np.zeros(1)
    
    t[0] = 0
    z[0] = [
        *z0,
        x0 + params.l*np.sin(params.control_theta),
        y0 - params.l*np.cos(params.control_theta)
    ]
    
    for i in range(step):
        one_step(z0, t0, params, i)
        
if __name__=="__main__":
    
    params = Params()
    
    step = len(params.vdes)
    
    x0dot = params.vdes[0]
    y0 = 1.2
    
    z0 = [0, x0dot, y0, 0]
    
    n_step(z0, params, step)