from matplotlib import pyplot as plt
from scipy.integrate import odeint

import numpy as np

class parameters:
    def __init__(self):
        # system parameters
        self.m = 1
        self.c = 1
        self.k = 1
        
        # control gains
        self.kp = 10
        self.kd = 1*(-self.c+2*np.sqrt(self.m*(self.k+self.kp)));

def smd_rhs(z,t, m, c, k, kp, kd):

    x = z[0];
    xdot = z[1];

    #orginal system
    #xddot = -k*x/m - c*xdot/m;

    #with control
    xddot = -( (k+kp)/m)*x-((c+kd)/m)*xdot;

    zdot = np.array([xdot, xddot]);

    return zdot

def plot(t, z):
    
    plt.figure(1)
    
    plt.plot(t,z[:,0])
    plt.xlabel("t")
    plt.ylabel("position")
    plt.title("Plot of position vs time")
    
    plt.show()
    

if __name__ == "__main__":
    params = parameters()
    
    x0, xdot0 = 0.5, 0
    t0, tend = 0, 20
    
    t = np.linspace(t0, tend, 101)
    z0 = np.array([x0, xdot0])
    
    # Extra arguments must be in a tuple.
    z = odeint(smd_rhs, z0, t, args=(params.m,params.c,params.k,params.kp,params.kd))

    plot(t, z)
    