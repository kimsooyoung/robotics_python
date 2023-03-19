from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.integrate import odeint

class parameters:
    def __init__(self):
        self.m = 1
        self.c = 1
        self.k = 1
        self.kp = 10
        self.kd = 1*(-self.c+2*np.sqrt(self.m*(self.k+self.kp)));

def smd_rhs(z,t,m,c,k,kp,kd):

    x = z[0];
    xdot = z[1];

    #orginal system
    #xddot = -k*x/m - c*xdot/m;

    #with control
    xddot = -( (k+kp)/m)*x-((c+kd)/m)*xdot;

    zdot = np.array([xdot, xddot]);

    return zdot

parms = parameters()
x0 = 0.5;
xdot0 = 0;
t0 = 0;
tend = 20;

t = np.linspace(t0, tend, 101)
z0 = np.array([x0, xdot0])
z = odeint(smd_rhs, z0, t, args=(parms.m,parms.c,parms.k,parms.kp,parms.kd))

plt.figure(1)
plt.plot(t,z[:,0])
plt.xlabel("t")
plt.ylabel("position")
plt.title("Plot of position vs time")
plt.show()
# plt.show(block=False)
# plt.pause(2)
# plt.close()
