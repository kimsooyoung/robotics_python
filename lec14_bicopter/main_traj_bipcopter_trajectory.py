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
        self.I = 0.1
        self.g = 9.81
        self.l = 0.2
        self.r = 0.05
        self.pause = 0.01
        self.fps =30

        self.kp_y = 300;
        self.kp_x = 300;
        self.kp_phi = 2500;
        self.kd_phi = 2*np.sqrt(self.kp_phi);

def figure8(x0,y0,h,t0,tN):

    N = int((tN-t0)/h) + 1;
    t = np.linspace(t0, tN,N)
    # print(len(t))
    T = t[N-1];
    A = 0.5;
    B = A;
    a = 2;
    b = 1;
    pi = np.pi
    tau = 2*pi*(-15*(t/T)**4+6*(t/T)**5+10*(t/T)**3);
    taudot = 2*pi*(-15*4*(1/T)*(t/T)**3+6*5*(1/T)*(t/T)**4+10*3*(1/T)*(t/T)**2);
    tauddot = 2*pi*(-15*4*3*(1/T)**2*(t/T)**2 + 6*5*4*(1/T)**2*(t/T)**3+10*3*2*(1/T)**2*(t/T));

    x = x0+A*sin(a*tau);
    y = y0+B*cos(b*tau);
    xdot =  A*a*cos(a*tau)*taudot;
    ydot = -B*b*sin(b*tau)*taudot;
    xddot = -A*a*a*sin(a*tau)*taudot+A*a*cos(a*tau)*tauddot;
    yddot = -B*b*b*sin(b*tau)*taudot-B*b*sin(b*tau)*tauddot;

    if (0): #code to check the curve
        plt.figure(1)
        plt.plot(x,y)
        plt.ylabel("y")
        plt.xlabel("x");
        plt.title("Plot of trajectory")
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    return t, x,y,xdot,ydot,xddot,yddot


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

    xx = max(z_interp[:,0]);
    yy = max(z_interp[:,1]);
    xxyy = max([xx,yy]);

    # #plot
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
             kp_y,kp_x,kp_phi,kd_phi,\
             x_ref,y_ref,xdot_ref, ydot_ref, \
             x2dot_ref,y2dot_ref, \
             m,I,g,l):

    kd_x = 2*np.sqrt(kp_x);
    kd_y = 2*np.sqrt(kp_y);

    us = m*(g + y2dot_ref - kp_y*(y - y_ref) - kd_y*(ydot - ydot_ref));
    phi_ref = -(1/g)*(x2dot_ref - kp_x*(x - x_ref) - kd_x*(xdot - xdot_ref));
    ud = -kp_phi*(phi - phi_ref) - kd_phi*(phidot);

    return us,ud

def bicopter_rhs(z,t,m,I,g,l,kp_y,kp_x,kp_phi,kd_phi, \
                x_ref,y_ref,xdot_ref, ydot_ref, \
                x2dot_ref, y2dot_ref):

    x = z[0];
    y = z[1];
    phi = z[2];
    xdot = z[3]
    ydot = z[4]
    phidot = z[5]

    [us,ud] = controller(x,y,phi,xdot,ydot,phidot, \
                 kp_y,kp_x,kp_phi,kd_phi,\
                 x_ref,y_ref,xdot_ref, ydot_ref, \
                 x2dot_ref,y2dot_ref, \
                 m,I,g,l)

    xddot = -(us/m)*sin(phi);
    yddot =  (us/m)*cos(phi) - g;
    phiddot = 0.5*l*ud/I;

    zdot = np.array([xdot, ydot, phidot, \
                     xddot, yddot, phiddot]);

    return zdot

#parameters
parms = parameters()

#center of the leminscate
x0_l = 0;
y0_l = 0;

# time and tspan
h = 0.005;
t0 = 0;
tN = 5;

# get reference velocity  %%%%
t, x_ref,y_ref,xdot_ref,ydot_ref, \
x2dot_ref,y2dot_ref  = figure8(x0_l,y0_l,h,t0,tN)

#initialization
x0 = x_ref[0];
y0 = y_ref[0]
phi0 = 0;
vx0 = 0;
vy0 = 0;
phidot0 = 0;
z0=np.array([x0, y0, phi0, vx0, vy0, phidot0]);

##state
N = len(t)
shape = (N,6) #z0 is size 6
z = np.zeros(shape)
us = np.zeros((N,1))
ud = np.zeros((N,1))
for i in np.arange(6):
    z[0,i] = z0[i]


for i in range(0,N-1):

    physical_parms = (parms.m,parms.I,parms.g,parms.l)
    control_parms = (parms.kp_y,parms.kp_x, parms.kp_phi, parms.kd_phi,\
                    x_ref[i],y_ref[i],xdot_ref[i],ydot_ref[i], \
                    x2dot_ref[i],y2dot_ref[i])
    all_parms = physical_parms + control_parms
    t_temp = np.array([t[i], t[i+1]])

    z_temp = odeint(bicopter_rhs, z0, t_temp, args=all_parms)

    us_temp,ud_temp  = controller(z0[0],z0[1],z0[2],z0[3],z0[4],z0[5], \
              parms.kp_y,parms.kp_x, parms.kp_phi, parms.kd_phi,\
              x_ref[i],y_ref[i],xdot_ref[i],ydot_ref[i], \
              x2dot_ref[i],y2dot_ref[i],
              parms.m,parms.I,parms.g,parms.l)

    z0 = np.array([z_temp[1,0], z_temp[1,1], z_temp[1,2], \
                   z_temp[1,3], z_temp[1,4], z_temp[1,5]])

    z[i+1,0] = z0[0]
    z[i+1,1] = z0[1]
    z[i+1,2] = z0[2]
    z[i+1,3] = z0[3]
    z[i+1,4] = z0[4]
    z[i+1,5] = z0[5]

    us[i+1,0] = us_temp
    ud[i+1,0] = ud_temp

animate(t,z,parms)

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,z[:,0])
plt.plot(t,x_ref,'r-.');
plt.ylabel("x")
plt.title("Plot of x vs. time")
plt.subplot(2,1,2)
plt.plot(t,z[:,1])
plt.plot(t,y_ref,'r-.');
plt.xlabel("time")
plt.show(block=False)
plt.pause(2)
plt.close()

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(t,z[:,3])
plt.plot(t,xdot_ref,'r-.');
plt.ylabel("xdot")
plt.title("Plot of velocity (x) vs. time")
plt.subplot(2,1,2)
plt.plot(t,z[:,4])
plt.plot(t,ydot_ref,'-.');
plt.ylabel("ydot")
plt.xlabel("time")
plt.show(block=False)
plt.pause(2)
plt.close()

plt.figure(3)
plt.subplot(2,1,1)
plt.plot(t,us[:,0])
plt.ylabel("Thrust (sum)")
plt.subplot(2,1,2)
plt.plot(t,ud[:,0])
plt.ylabel("Thrust (diff)")
plt.xlabel("time")
plt.show(block=False)
plt.pause(4)
plt.close()
