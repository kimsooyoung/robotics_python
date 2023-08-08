from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.integrate import odeint

class parameters:
    def __init__(self):
        self.m1 = 1
        self.m2 = 1
        self.k1 = 2
        self.k2 = 3

        self.pause = 0.01
        self.fps =20

def spring_mass_rhs(x,t,m1,m2,k1,k2):

    A = np.array([
                [0,0,1,0],
                [0,0,0,1],
                [-(k1/m1+k2/m1), k2/m1, 0, 0],
                [k2/m2, -k2/m2, 0, 0]
                ])

    B = np.array([
                 [0,0],
                 [0,0],
                 [-1/m1, 0],
                 [1/m2,1/m2]
                 ])

    #uncontrolled
    #u = np.array([0,0])

    #lqr control (copy pasted from spring_mass.py)
    K= np.array([[-5.41083775,  2.35057376, -9.38138674,  4.56991175],
                  [ 2.54031466,  8.96500756,  5.28127709,  9.85118885]]);
    u = -K.dot(x)


    xdot = A.dot(x)+B.dot(u)


    return xdot

parms = parameters()
x0 = np.array([0.5,0,0,0])
t0 = 0;
tend = 20;

t = np.linspace(t0, tend, 101)
x = odeint(spring_mass_rhs, x0, t, args=(parms.m1,parms.m2,parms.k1,parms.k2))

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,x[:,0],'r-.')
plt.plot(t,x[:,1],'b');
plt.ylabel("position")
plt.subplot(2,1,2)
plt.plot(t,x[:,2],'r-.')
plt.plot(t,x[:,3],'b');
plt.ylabel("velocity")
plt.xlabel("t")
plt.show(block=False)
plt.pause(5)
plt.close()
