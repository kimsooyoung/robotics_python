import numpy as np

from scipy import interpolate
from scipy.optimize import fsolve
from scipy.integrate import odeint


from derive_fourlinkchain import eq_auto_generation
from fourlinkchain_rhs import fourlinkchain_rhs
from get_reference import get_reference
from visualize import animate, plot_result

class parameters:
    def __init__(self):
        
        # self.leg = "atrias"
        # self.leg = "minitaur"
        self.leg = "digit"
        
        self.show_phase = False
        
        self.m1 = 1; self.m2 = 1; self.m3 = 1; self.m4 = 1;
        self.I1 = 0.1; self.I2 = 0.1; self.I3 = 0.1; self.I4 = 0.1;
        
        if self.leg == "minitaur":
            self.l1 = 1; self.l2 = 2; self.l3 = 1; self.l4 = 2;
        elif self.leg == "atrias" or self.leg == "digit":
            self.l1 = 1; self.l2 = 2; self.l3 = 2; self.l4 = 1;
        
        self.lx = 0; self.ly = 0;
        self.g = 9.81
        
        self.Kp1 = 0*100; self.Kp2 = 0*100
        self.Kd1 = 2 * np.sqrt(self.Kp1); self.Kd2 = 2 * np.sqrt(self.Kp2)
        
        self.l_init = 0.9*(self.l1 + self.l2)
        self.alpha_init = -0.5
        
        self.l_mid = 0.6*(self.l1 + self.l2)
        self.alpha_mid = 0
        
        self.l_final = 0.9*(self.l1 + self.l2)
        self.alpha_final = 0.5
        
        self.t_end = 2.0
        
        self.pause = 0.02
        self.fps = 20
        
def cos(angle):
    return np.cos(angle)

def sin(angle):
    return np.sin(angle);

def interpolation(t, z, params):

    #interpolation
    t_interp = np.arange(t[0], t[len(t)-1], 1/params.fps)
    # [rows, cols] = np.shape(z)
    [cols, rows] = np.shape(z)
    z_interp = np.zeros((len(t_interp), rows))

    for i in range(0, rows):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    return t_interp, z_interp

def position_last_link_tip(z, params):
    
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4
    lx, ly = params.lx, params.ly
    
    q1, q2, q3, q4 = z
    
    del_x = l2*sin(q1 + q2) - lx - l4*sin(q3 + q4) + l1*sin(q1) - l3*sin(q3)
    del_y = l4*cos(q3 + q4) - l2*cos(q1 + q2) - ly - l1*cos(q1) + l3*cos(q3)
    
    return del_x, del_y, 0, 0

def velocity_last_link_tip(z, params, q_star):
    
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4
    q1, q2, q3, q4 = q_star
    u1, u2, u3, u4 = z
    
    del_vx = u1*(l2*cos(q1 + q2) + l1*cos(q1)) - u3*(l4*cos(q3 + q4) + l3*cos(q3)) + l2*u2*cos(q1 + q2) - l4*u4*cos(q3 + q4);
    del_vy = u1*(l2*sin(q1 + q2) + l1*sin(q1)) - u3*(l4*sin(q3 + q4) + l3*sin(q3)) + l2*u2*sin(q1 + q2) - l4*u4*sin(q3 + q4);

    return del_vx, del_vy, 0, 0


if __name__=="__main__":

    params = parameters()
    show_phase = params.show_phase
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4
    
    eq_auto_generation(params.leg)
    
    # q2 here means second controllerable angle
    q_ini, t_ref, q1_refs, q2_refs = get_reference(params)
    q1, q2, q3, q4 = q_ini[0], q_ini[1], q_ini[2], q_ini[3]
    u1, u2, u3, u4 = 0, 0, 0, 0
    
    ### Use ode45 to do simulation ###
    z0 = np.array([
        q1, u1, 
        q2, u2, 
        q3, u3, 
        q4, u4
    ])

    z_result = np.zeros((len(t_ref)-1,8))
    
    for i in range(len(t_ref)-1):
        t = np.array([t_ref[i], t_ref[i+1]])
        q1_ref = q1_refs[i,:]
        q2_ref = q2_refs[i,:]
        
        z = odeint(
            fourlinkchain_rhs, z0, t, args=(params, q1_ref, q2_ref),
            rtol=1e-9, atol=1e-9, mxstep=9000
        )
        z0 = z[-1]
        z_result[i] = z0
    
    t_interp, z_interp = interpolation(t_ref[1:], z_result, params)
    animate(t_interp, z_interp, params)
    plot_result(t_interp, z_interp, t_ref, q1_refs, q2_refs, z_result, params)
    
    
