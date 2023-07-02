from matplotlib import pyplot as plt
import numpy as np

from scipy import interpolate
from scipy.integrate import odeint

from derive_nlink import derive_nlink

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

if __name__=="__main__":

    params = parameters()
    # derive_nlink(params.dof, params.method)
    
    from nlink_animate import nlink_animate
    from nlink_plot import nlink_plot
    from nlink_rhs import nlink_rhs

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
        print(f"{end - start:.5f} sec") # 0.61312 sec
    except Exception as e:
        print(e)
    finally:
        t_interp, z_interp = interpolation(t, z, params)
        # nlink_animate(t_interp, z_interp, params)
        nlink_plot(t_interp, z_interp, params)
        print("done")
