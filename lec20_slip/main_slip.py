import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# TODO:
# 1. flight eom
#    - contact event
#    - apex event
# 2. stance eom
#    - release event
#
# 3. one bounce
#
# 4. animation

class Params:
    def __init__(self):
        self.g = 9.81
        self.ground = 0.0
        self.l = 1
        self.m = 1
        
        # sprint stiffness
        self.k = 100
        # fixed angle
        self.theta = 10 * np.pi / 180
        
        self.pause = 0.005
        self.fps = 10

def flight(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    
    return [x_dot, 0, y_dot, -g]

def contact(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    # contact event
    return y - l0 * np.cos(theta)

def apex(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    return y_dot
# apex.direction = 0
# apex.terminal = True

def stance(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    
    l = np.sqrt(x**2 + y**2)
    F_spring = k * (l0 - l)
    Fx_spring = F_spring * x / l
    Fy_spring = F_spring * y / l
    Fy_gravity = m*g
    
    x_dd = (Fx_spring) / m
    y_dd = (Fy_spring - Fy_gravity) / m
    
    return [x_dot, x_dd, y_dot, y_dd]

def release(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    l = np.sqrt(x**2 + y**2)
    
    return l - l0
# release.direction = +1
# release.terminal = True

def onestep(z0, t0, params):
    
    dt = 5
    x, x_d, y, y_d = z0
    m, g, k = params.m, params.g, params.k
    l0, theta = params.l, params.theta
    
    # z_output = [x, x_dot, y, y_dot, x_foot, y_foot]
    z_output = np.zeros((1,6))
    z_output[0] = [*z0, x+l0*np.sin(theta), y-l0*np.cos(theta)]
    
    print(z_output)
    
    contact.direction = -1
    contact.terminal = True
    
    ts = np.linspace(t0, t0+dt, 1001)
    # def contact(t, z, l0, theta):
    contact_sol = solve_ivp(
        flight, [t0, t0+dt], z0, method='RK45', t_eval=np.linspace(t0, t0+dt, 1001),
        dense_output=True, events=contact, atol = 1e-13, rtol = 1e-12, 
        args=(m, g, l0, k, theta)
    )
    
    t_temp = contact_sol.t
    m, n = np.shape(contact_sol.y)
    z_temp = contact_sol.y.T
    
    t0, z0 = t_temp[-1], z_temp[-1]
    
    ts = np.linspace(t0, t0+dt, 1001)
    # def stance(t, z, m, g, l0, k, theta)
    release_sol = solve_ivp(
        stance, [t0, t0+dt], z0, method='RK45', t_eval=np.linspace(t0, t0+dt, 1001),
        dense_output=True, events=contact, atol = 1e-13, rtol = 1e-12, 
        args=(m, g, l0, k, theta)
    )
    
    


def n_step(zstar,params,steps):
    
    # x0 = 0; x0dot = z0(1);  
    # y0 = z0(2); y0dot = 0;
    
    z0 = zstar
    t0 = 0
    
    for i in range(steps):
        onestep(z0, t0, params)
        


if __name__=="__main__":
    
    params = Params()

    x, x_d, y, y_d = 0, 1, 1.2, 0
    z0 = np.array([x, x_d, y, y_d])
    
    n_step(z0, params, 1)
    