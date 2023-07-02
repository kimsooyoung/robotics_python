from matplotlib import pyplot as plt
import numpy as np

from scipy import interpolate
from scipy.integrate import odeint

from cython_dynamics import nlink_rhs_cython

class parameters:
    
    def __init__(self):
        
        self.dof = 3
        self.method = "zeroref"
        
        for i in range(self.dof):
            setattr(self, f"m{i+1}", 1)
            setattr(self, f"l{i+1}", 1)
            setattr(self, f"I{i+1}", 0.1)
            setattr(self, f"c{i+1}", 0.5)
            
        self.g = 9.81
        
        self.pause = 0.02
        self.fps = 20

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

def nlink_rhs(z, t, params):

    q_0, u_0 = z[0], z[1] 
    q_1, u_1 = z[2], z[3] 
    q_2, u_2 = z[4], z[5] 
    
    m_0 = params.m1; I_0 = params.I1
    c_0 = params.c1; l_0 = params.l1
    m_1 = params.m2; I_1 = params.I2
    c_1 = params.c2; l_1 = params.l2
    m_2 = params.m3; I_2 = params.I3
    c_2 = params.c3; l_2 = params.l3
    g = params.g
    
    params_arr = np.array([
        m_0, I_0, c_0, l_0,
        m_1, I_1, c_1, l_1,
        m_2, I_2, c_2, l_2,
        g
    ])
    
    M, C, G = nlink_rhs_cython.nlink_rhs(z, params_arr)
    b = -C - G
    
    # print(f"M = {M}")
    # print(f"C = {C}")
    # print(f"G = {G}")
    # print(f"b = {b}")
    
    x = np.linalg.solve(M, b)
    
    # print(f"x = {x}")

    output = np.array([
        u_0, x[0,0],
        u_1, x[1,0],
        u_2, x[2,0]
    ])

    return output 
    

if __name__=="__main__":

    params = parameters()
    # derive_nlink(params.dof, params.method)
    
    from nlink_animate import nlink_animate
    from nlink_plot import nlink_plot

    z = None
    total_time = 5
    t = np.linspace(0, total_time, 100*total_time)
    
    ### Use ode45 to do simulation ###
    z0 = np.array([0.0] * (2*params.dof))
    z0[0] = np.pi/2
    
    try:
        import time 
        
        start = time.time()
        z = odeint(
            nlink_rhs, z0, t, args=(params,),
            rtol=1e-12, atol=1e-12
        )
        end = time.time()
        print(f"{end - start:.5f} sec") # 0.11163 sec
    except Exception as e:
        print(e)
    finally:
        t_interp, z_interp = interpolation(t, z, params)
        # nlink_animate(t_interp, z_interp, params)
        nlink_plot(t_interp, z_interp, params)
        print("done")
