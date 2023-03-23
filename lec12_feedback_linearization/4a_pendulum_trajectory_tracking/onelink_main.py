from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.integrate import odeint
from scipy import interpolate

def cos(angle):
    return np.cos(angle)

def sin(angle):
    return np.sin(angle);

class parameters:
    def __init__(self):
        self.m1 = 1
        self.I1 = 0.5
        self.g = 10
        self.l1 = 1
        self.pause = 0.01
        self.fps =20

        self.kp1 = 200
        self.kd1 = 2*np.sqrt(self.kp1)

def animate(t,z,parms):
    #interpolation
    t_interp = np.arange(t[0],t[len(t)-1],1/parms.fps)
    [m,n] = np.shape(z)
    shape = (len(t_interp),n)
    z_interp = np.zeros(shape)

    for i in range(0,n-1):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    l1 = parms.l1

    #plot
    for i in range(0,len(t_interp)):
        theta1 = z_interp[i,0];
        O = np.array([0, 0])
        P = np.array([l1*cos(theta1), l1*sin(theta1)])

        pend1, = plt.plot([O[0], P[0]],[O[1], P[1]],linewidth=5, color='red')

        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.gca().set_aspect('equal')

        plt.pause(parms.pause)
        pend1.remove()

    plt.close()

def control(theta1,theta1dot,kp1,kd1,theta1_ref,theta1dot_ref,theta1ddot_ref,m1,g,l1,I1):
    T1 = (I1+m1*l1*l1/4)*(theta1ddot_ref-kp1*(theta1-theta1_ref)-kd1*(theta1dot-theta1dot_ref)) + 0.5*m1*g*l1*cos(theta1)
    return T1

def onelink_rhs(z,t,m1,I1,l1,g,kp1,kd1,theta1_ref,theta1dot_ref,theta1ddot_ref,T1_disturb):

    theta1 = z[0];
    theta1dot = z[1];

    T1 = control(theta1,theta1dot,kp1,kd1,theta1_ref,theta1dot_ref,theta1ddot_ref,m1,g,l1,I1)-T1_disturb
    theta1ddot = (1/( I1+(m1*l1*l1/4) ))*(T1 - 0.5*m1*g*l1*cos(theta1));

    zdot = np.array([theta1dot, theta1ddot]);

    return zdot

def plot(t, z, theta_ref, thetadot_ref, T):

    plt.figure(1)

    plt.subplot(3,1,1)
    plt.plot(t,z[:,0])
    plt.plot(t,theta_ref,'r+');
    plt.ylabel("theta1")
    plt.title("Plot of position, velocity, and Torque vs. time")
    
    plt.subplot(3,1,2)
    plt.plot(t,z[:,1])
    plt.plot(t,thetadot_ref,'r+');
    plt.ylabel("theta1dot")
    
    plt.subplot(3,1,3)
    plt.plot(t,T[:,0])
    plt.xlabel("t")
    plt.ylabel("Torque")

    plt.show()

if __name__=="__main__":
#parameters
    parms = parameters()

    #initialization
    theta1, theta1dot = 0, 0

    # disturbances
    T1_mean, T1_dev = 0.0, 40 * 0
    theta1_mean, theta1_dev = 0.0, 0.1 * 0.0
    theta1dot_mean, theta1dot_dev = 0.0, 0.5 * 0.0

    #set the time
    t1_0, t1_N = 0, 1.5
    t2_0, t2_N = 1.5, 3

    #time
    h = 0.005;
    N1 = int((t1_N-t1_0)/h) + 1;
    time1 = np.linspace(t1_0, t1_N,N1)
    N2 = int((t2_N-t2_0)/h) + 1;
    time2 = np.linspace(t2_0, t2_N,N2)
    
    t = np.concatenate((time1,time2[1:]))

    pi = np.pi;

    a10 =  0
    a11 =  0
    a12 =  0.666666666666667*pi
    a13 =  -0.296296296296296*pi
    a20 =  -2.0*pi
    a21 =  4.0*pi
    a22 =  -2.0*pi
    a23 =  0.296296296296296*pi

    thetaA = a10 + a11*time1 + a12*time1**2 + a13*time1**3;
    thetaAdot = a11 + 2*a12*time1 + 3*a13*time1**2;
    thetaAddot = 2*a12 + 6*a13*time1;

    thetaB = a20 + a21*time2 + a22*time2**2 + a23*time2**3;
    thetaBdot = a21 + 2*a22*time2 + 3*a23*time2**2;
    thetaBddot = 2*a22 + 6*a23*time2;

    theta_ref = np.concatenate((thetaA,thetaB[1:]))
    thetadot_ref = np.concatenate((thetaAdot,thetaBdot[1:]))
    thetaddot_ref = np.concatenate((thetaAddot,thetaBddot[1:]))

    #state
    z = np.zeros( (len(t), 2) )
    T = np.zeros( (len(t),1) )

    z0 = np.array([theta1, theta1dot])
    z[0] = z0

    for i in range(len(t)-1):
        theta1_ref, theta1dot_ref, theta1ddot_ref = theta_ref[i], thetadot_ref[i], thetaddot_ref[i]
        physical_parms = (parms.m1,parms.I1,parms.l1,parms.g)
        T1_disturb = np.random.normal(T1_mean,T1_dev)

        control_parms = (parms.kp1,parms.kd1,theta1_ref,theta1dot_ref,theta1ddot_ref, T1_disturb)
        all_parms = physical_parms + control_parms

        t_temp = np.array([t[i], t[i+1]])
        z_temp = odeint(onelink_rhs, z0, t_temp, args=all_parms)
        T_temp  = control(z0[0],z0[1],parms.kp1,parms.kd1,theta1_ref,theta1dot_ref,theta1ddot_ref,parms.m1,parms.g,parms.l1,parms.I1)

        z0 = np.array([z_temp[1,0]+np.random.normal(theta1_mean,theta1_dev), \
                    z_temp[1,1]+np.random.normal(theta1dot_mean,theta1dot_dev)])

        z[i+1] = z0
        T[i+1] = T_temp


    animate(t,z,parms)
    plot(t, z, theta_ref, thetadot_ref, T)