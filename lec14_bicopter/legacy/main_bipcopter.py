from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.integrate import odeint
from scipy import interpolate
from scipy.optimize import fsolve

def cos(angle):
    return np.cos(angle)

def sin(angle):
    return np.sin(angle);

class parameters:
    def __init__(self):
        self.m = 1
        self.g = 9.81
        self.l = 0.2
        self.r = 0.05
        
        self.I = self.m*self.l**2/12
        self.pause = 0.01
        self.fps =30

def animate(t,z,parms):
    #interpolation
    t_interp = np.arange(t[0],t[len(t)-1],1/parms.fps)
    [m,n] = np.shape(z)
    shape = (len(t_interp),n)
    z_interp = np.zeros(shape)

    for i in range(0,n-1):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    l = parms.l
    r = parms.r

    # xx = max(abs(z_interp[:,0]));
    # yy = max(abs(z_interp[:,1]));
    # xxyy = max([xx,yy]);
    xxyy = 1

    # plot
    for i in range(0,len(t_interp)):
        x = z_interp[i,0];
        y = z_interp[i,1];
        phi = z_interp[i,2]

        R = np.array([[cos(phi), -sin(phi)],
                     [sin(phi), cos(phi)]])
        middle = np.array([x,y])

        drone_left = np.add(middle,R.dot(np.array([-0.5*l,0])))
        axle_left = np.add(middle,R.dot(np.array([-0.5*l,0.1])))
        prop_left1 = np.add(middle, \
                          np.add( \
                          R.dot(np.array([-0.5*l,0.05])), R.dot(np.array([0.5*r,0.])) \
                          ), \
                          )
        prop_left2 = np.add(middle, \
                          np.add( \
                          R.dot(np.array([-0.5*l,0.05])), R.dot(np.array([-0.5*r,0.])) \
                          ), \
                          )

        drone_right = np.add(middle,R.dot(np.array([0.5*l,0])))
        axle_right = np.add(middle,R.dot(np.array([0.5*l,0.1])))
        prop_right1 = np.add(middle, \
                          np.add( \
                          R.dot(np.array([0.5*l,0.05])), R.dot(np.array([0.5*r,0.])) \
                          ), \
                          )
        prop_right2 = np.add(middle, \
                          np.add( \
                          R.dot(np.array([0.5*l,0.05])), R.dot(np.array([-0.5*r,0.])) \
                          ), \
                          )

        drone, = plt.plot([drone_left[0],drone_right[0]], \
                           [drone_left[1],drone_right[1]],linewidth=5, color='red');
        prop_left_stand, = plt.plot([drone_left[0],axle_left[0]], \
                           [drone_left[1],axle_left[1]],linewidth=5, color='green');
        prop_left, = plt.plot([prop_left1[0],prop_left2[0]], \
                           [prop_left1[1],prop_left2[1]],linewidth=5, color='blue');
        prop_right_stand, = plt.plot([drone_right[0],axle_right[0]], \
                           [drone_right[1],axle_right[1]],linewidth=5, color='green');
        prop_right, = plt.plot([prop_right1[0],prop_right2[0]], \
                           [prop_right1[1],prop_right2[1]],linewidth=5, color='blue');

        endEff, = plt.plot(x,y,color='black',marker='o',markersize=2)

        plt.xlim(-xxyy-0.1,xxyy+0.1)
        plt.ylim(-xxyy-0.1,xxyy+0.1)
        plt.gca().set_aspect('equal')

        plt.pause(parms.pause)
        drone.remove()
        prop_left_stand.remove()
        prop_left.remove()
        prop_right_stand.remove()
        prop_right.remove()

    plt.close()

def controller(x,y,phi,xdot,ydot,phidot, \
             m,I,g,l):

    us = m*g+0.01
    ud = 0

    # phi_ref = -50*(x-0)
    # ud = -100*(phi-0.2)-10*phidot

    return us,ud

def bicopter_rhs(z,t,m,I,g,l):
    
    x, y, phi, xdot, ydot, phidot = z
    
    [us,ud] = controller(x,y,phi,xdot,ydot,phidot, m,I,g,l)

    xddot = -(us/m)*sin(phi);
    yddot =  (us/m)*cos(phi) - g;
    phiddot = 0.5*l*ud/I;

    zdot = np.array([xdot, ydot, phidot, \
                     xddot, yddot, phiddot]);

    return zdot


if __name__=="__main__":
    #parameters
    parms = parameters()

    # time and tspan
    h = 0.005;
    t0 = 0;
    tN = 4;

    N = int((tN-t0)/h) + 1;
    t = np.linspace(t0, tN,N)

    #initialization
    x0, y0, phi0, vx0, vy0, phidot0 = 0, 0, -0.1, 0, 0, 0
    z0=np.array([x0, y0, phi0, vx0, vy0, phidot0]);

    z = np.zeros((N,6))
    us = np.zeros((N,1))
    ud = np.zeros((N,1))
    z[0] = z0

    for i in range(N-1):

        physical_parms = (parms.m,parms.I,parms.g,parms.l)
        all_parms = physical_parms

        t_temp = np.array([t[i], t[i+1]])
        z_temp = odeint(bicopter_rhs, z0, t_temp, args=all_parms)
        us_temp,ud_temp  = controller(z0[0],z0[1],z0[2],z0[3],z0[4],z0[5], \
                parms.m,parms.I,parms.g,parms.l)

        z0 = np.array([z_temp[1,0], z_temp[1,1], z_temp[1,2], \
                    z_temp[1,3], z_temp[1,4], z_temp[1,5]])

        z[i+1] = z0
        us[i+1,0] = us_temp
        ud[i+1,0] = ud_temp

    animate(t,z,parms)
