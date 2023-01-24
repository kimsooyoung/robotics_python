from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import interpolate
from scipy.integrate import odeint

class parameters:
    def __init__(self):
        self.g = 9.81
        self.m = 1
        self.c = 0.47
        self.pause = 0.01
        self.fps = 10

def animate(t,z,parms):
    #interpolation
    t_interp = np.arange(t[0],t[len(t)-1],1/parms.fps)
    [m,n] = np.shape(z)
    shape = (len(t_interp),n)
    z_interp = np.zeros(shape)

    for i in range(0,n-1):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    #plot
    for i in range(0,len(t_interp)):
        traj, = plt.plot(z_interp[0:i,0],z_interp[0:i,2],color='red');
        prj, =  plt.plot(z_interp[i,0],z_interp[i,2],color='red',marker='o');

        plt.xlim(min(z[:,0]-1),max(z[:,0]+1))
        plt.ylim(min(z[:,2]-1),max(z[:,2]+1))
        # plt.gca().set_aspect('equal')
        plt.pause(parms.pause)
        traj.remove()
        prj.remove()

    plt.close()

def projectile(z, t, m,g,c):

    xdot = z[1];
    ydot = z[3];
    v = np.sqrt(xdot**2+ydot**2);

    #%%%% drag is prop to v^2
    dragX = c*v*xdot;
    dragY = c*v*ydot;

    #%%%% net acceleration %%%
    ax = 0-(dragX/m); #xddot
    ay = -g-(dragY/m); #yddot

    dzdt = np.array([xdot, ax, ydot, ay]);
    return dzdt

parms = parameters()
x0 = 0;
x0dot = 100;
y0 = 0;
y0dot = x0dot*math.tan(math.pi/3);
t0 = 0;
tend = 5;

t = np.linspace(t0, tend, 101)
z0 = np.array([x0, x0dot, y0, y0dot])
z = odeint(projectile, z0, t, args=(parms.m,parms.g,parms.c))

animate(t,z,parms)
