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
        self.m2 = 1
        self.I2 = 0.5
        self.g = 10
        self.c1 = 0.5
        self.c2 = 0.5
        self.l = 1
        self.pause = 0.01
        self.fps =30

        self.kp1 = 100
        self.kd1 = 2*np.sqrt(self.kp1)
        self.kp2 = 100
        self.kd2 = 2*np.sqrt(self.kp2)

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
    c1 = parms.c1
    c2 = parms.c2

    # #plot
    for i in range(0,len(t_interp)):
        theta1 = z_interp[i,0];
        theta2 = z_interp[i,2];
        O = np.array([0, 0])
        P = np.array([l*cos(theta1), l*sin(theta1)])
        Q = P + np.array([l*cos(theta1+theta2),l*sin(theta1+theta2)])
        G1 = np.array([c1*cos(theta1), c1*sin(theta1)])
        G2 = P + np.array([c2*cos(theta1+theta2),c2*sin(theta1+theta2)])

        pend1, = plt.plot([O[0], P[0]],[O[1], P[1]],linewidth=5, color='red')
        pend2, = plt.plot([P[0], Q[0]],[P[1], Q[1]],linewidth=5, color='green')
        com1, = plt.plot(G1[0],G1[1],color='black',marker='o',markersize=10)
        com2, = plt.plot(G2[0],G2[1],color='black',marker='o',markersize=10)
        endEff, = plt.plot(Q[0],Q[1],color='black',marker='o',markersize=5)

        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.gca().set_aspect('equal')


        plt.pause(parms.pause)
        pend1.remove()
        pend2.remove()
        com1.remove()
        com2.remove()

    plt.close()

def control(theta1,theta1dot,theta2,theta2dot, \
            kp1,kd1,kp2,kd2, \
            theta1ref,theta1dotref,theta1ddotref, \
            theta2ref,theta2dotref,theta2ddotref, \
            m1,I1,c1,m2,I2,c2,l,g):

    #pd control
    #T1 = -kp1*(theta1-theta1ref)-kd1*(theta1dot-theta1dotref);
    #T2 = -kp2*(theta2-theta2ref)-kd2*(theta2dot-theta2dotref)

    #control partitioning
    M11 =  1.0*I1 + 1.0*I2 + c1**2*m1 + m2*(c2**2 + 2*c2*l*cos(theta2) + l**2)
    M12 =  1.0*I2 + c2*m2*(c2 + l*cos(theta2))
    M21 =  1.0*I2 + c2*m2*(c2 + l*cos(theta2))
    M22 =  1.0*I2 + c2**2*m2

    C1 =  -c2*l*m2*theta2dot*(2.0*theta1dot + 1.0*theta2dot)*sin(theta2)
    C2 =  c2*l*m2*theta1dot**2*sin(theta2)

    G1 =  g*(c1*m1*cos(theta1) + m2*(c2*cos(theta1 + theta2) + l*cos(theta1)))
    G2 =  c2*g*m2*cos(theta1 + theta2)

    M = np.array([  [M11, M12], [M21,M22]  ]);
    CG = np.array([C1+G1,C2+G2])

    Kp = np.array([ [kp1, 0], [0, kp2] ]);
    Kd = np.array([ [kd1, 0], [0, kd2] ]);

    theta = np.array([theta1, theta2])
    thetadot = np.array([theta1dot, theta2dot])
    theta_ref = np.array([theta1ref, theta2ref])
    thetadot_ref = np.array([theta1dotref, theta2dotref])
    thetaddot_ref = np.array([theta1ddotref,theta2ddotref]);

    T = M.dot((thetaddot_ref-Kp.dot(theta-theta_ref)-Kd.dot(thetadot-thetadot_ref)))+CG;

    T1 = T[0]
    T2 = T[1]
    return T1,T2

def twolink_rhs(z,t,m1,I1,c1,m2,I2,c2,l,g,kp1,kd1,kp2,kd2, \
                theta1ref,theta1dotref,theta1ddotref, \
                theta2ref,theta2dotref,theta2ddotref, \
                T1_disturb,T2_disturb):

    theta1 = z[0];
    theta1dot = z[1];
    theta2 = z[2];
    theta2dot = z[3]

    [T1,T2] = control(theta1,theta1dot,theta2,theta2dot, \
                 kp1,kd1,kp2, kd2, \
                 theta1ref,theta1dotref,theta1ddotref, \
                 theta2ref,theta2dotref,theta2ddotref, \
                 m1,I1,c1,m2,I2,c2,l,g)
    T1 = T1 - T1_disturb;
    T2 = T2 - T2_disturb;

    M11 =  1.0*I1 + 1.0*I2 + c1**2*m1 + m2*(c2**2 + 2*c2*l*cos(theta2) + l**2)
    M12 =  1.0*I2 + c2*m2*(c2 + l*cos(theta2))
    M21 =  1.0*I2 + c2*m2*(c2 + l*cos(theta2))
    M22 =  1.0*I2 + c2**2*m2

    C1 =  -c2*l*m2*theta2dot*(2.0*theta1dot + 1.0*theta2dot)*sin(theta2)
    C2 =  c2*l*m2*theta1dot**2*sin(theta2)

    G1 =  g*(c1*m1*cos(theta1) + m2*(c2*cos(theta1 + theta2) + l*cos(theta1)))-T1
    G2 =  c2*g*m2*cos(theta1 + theta2)-T2


    A = np.array([[M11, M12], [M21,M22]]);
    b = -np.array([C1+G1,C2+G2])
    invA = np.linalg.inv(A)
    thetaddot = invA.dot(b)
    theta1ddot = thetaddot[0]
    theta2ddot = thetaddot[1]

    zdot = np.array([theta1dot, theta1ddot, theta2dot, theta2ddot]);

    return zdot

def plot(t,z,theta1_ref,theta2_ref):

    plt.figure(1)
    
    plt.subplot(2,1,1)
    plt.plot(t,z[:,0])
    plt.plot(t,theta1_ref,'r-.');
    plt.ylabel("theta1")
    plt.title("Plot of position vs. time")
    
    plt.subplot(2,1,2)
    plt.plot(t,z[:,2])
    plt.plot(t,theta2_ref,'r-.');
    plt.ylabel("theta2")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(t,z[:,1])
    plt.plot(t,theta1dot_ref,'r-.');
    plt.ylabel("theta1dot")
    plt.title("Plot of velocity vs. time")
    
    plt.subplot(2,1,2)
    plt.plot(t,z[:,3])
    plt.plot(t,theta2dot_ref,'-.');
    plt.ylabel("theta2dot")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    plt.figure(3)
    plt.subplot(2,1,1)
    plt.plot(t,T1[:,0])
    plt.ylabel("Torque 1")
    
    plt.subplot(2,1,2)
    plt.plot(t,T2[:,0])
    plt.ylabel("Torque 2")
    plt.xlabel("time")
    plt.show(block=False)
    plt.pause(4)
    plt.close()

if __name__=="__main__":
    #parameters
    parms = parameters()

    h = 0.005;
    pi = np.pi;

    # disturbances
    T1_mean = 0;
    T1_dev = 40*0;
    T2_mean = 0;
    T2_dev = 40*0;
    theta1_mean = 0;
    theta1_dev = 0.0;
    theta2_mean = 0;
    theta2_dev = 0.0;
    theta1dot_mean = 0
    theta1dot_dev = 0.5*0
    theta2dot_mean = 0
    theta2dot_dev = 0.5*0

    #set the time
    t1_0 =  0
    t1_N =  3.0
    a10 =  -0.5*pi - 0.5
    a11 =  0
    a12 =  0.333333333333333
    a13 =  -0.0740740740740741
    
    N1 = int((t1_N-t1_0)/h) + 1;
    time1 = np.linspace(t1_0, t1_N,N1)
    theta1_ref =a10 + a11*time1 + a12*time1**2 + a13*time1**3;
    theta1dot_ref =a11 + 2*a12*time1 + 3*a13*time1**2;
    theta1ddot_ref =2*a12 + 6*a13*time1;

    #set the time
    t1_0 =  0
    t1_N =  1.5
    t2_0 =  1.5
    t2_N =  3

    #time
    N1 = int((t1_N-t1_0)/h) + 1;
    time1 = np.linspace(t1_0, t1_N,N1)
    N2 = int((t2_N-t2_0)/h) + 1;
    time2 = np.linspace(t2_0, t2_N,N2)
    mm = len(time2)
    t = np.concatenate((time1,time2[1:mm]))

    pi = np.pi;
    a10 =  0
    a11 =  0
    a12 =  0.666666666666667 - 0.666666666666667*pi
    a13 =  -0.296296296296296 + 0.296296296296296*pi
    a20 =  -2.0 + 2.0*pi
    a21 =  4.0 - 4.0*pi
    a22 =  -2.0 + 2.0*pi
    a23 =  0.296296296296296 - 0.296296296296296*pi

    thetaA =a10 + a11*time1 + a12*time1**2 + a13*time1**3;
    thetaAdot =a11 + 2*a12*time1 + 3*a13*time1**2;
    thetaAddot =2*a12 + 6*a13*time1;

    thetaB =a20 + a21*time2 + a22*time2**2 + a23*time2**3;
    thetaBdot =a21 + 2*a22*time2 + 3*a23*time2**2;
    thetaBddot =2*a22 + 6*a23*time2;

    mm = len(thetaB)
    theta2_ref = np.concatenate((thetaA,thetaB[1:mm]))
    theta2dot_ref = np.concatenate((thetaAdot,thetaBdot[1:mm]))
    theta2ddot_ref = np.concatenate((thetaAddot,thetaBddot[1:mm]))

    #initialization
    theta1 = theta1_ref[0];
    theta1dot = theta1dot_ref[0];
    theta2 = theta2_ref[0]
    theta2dot = theta2dot_ref[0]

    #state
    N = len(t)
    shape = (N,4) #2 is for theta1 and theta2 and their rates, change according to the system
    z = np.zeros(shape)
    T1 = np.zeros((N,1))
    T2 = np.zeros((N,1))
    z0 = np.array([theta1, theta1dot, theta2, theta2dot])
    z[0] = z0
    
    for i in range(0,N-1):
        theta1ref, theta1dotref, theta1ddotref = theta1_ref[i], theta1dot_ref[i], theta1ddot_ref[i]
        theta2ref, theta2dotref, theta2ddotref = theta2_ref[i], theta2dot_ref[i], theta2ddot_ref[i]
        
        T1_disturb = np.random.normal(T1_mean,T1_dev)
        T2_disturb = np.random.normal(T2_mean,T2_dev)
        physical_parms = (parms.m1,parms.I1,parms.c1,parms.m2,parms.I2,parms.c2,parms.l,parms.g)
        control_parms = (parms.kp1,parms.kd1, parms.kp2, parms.kd2,\
                        theta1ref,theta1dotref,theta1ddotref, \
                        theta2ref,theta2dotref,theta2ddotref, \
                        T1_disturb,T2_disturb)
        all_parms = physical_parms + control_parms
        
        t_temp = np.array([t[i], t[i+1]])
        z_temp = odeint(twolink_rhs, z0, t_temp, args=all_parms)
        
        T1_temp, T2_temp  = control(z0[0],z0[1],z0[2],z0[3], \
                parms.kp1,parms.kd1, parms.kp2,parms.kd2,\
                theta1ref,theta1dotref,theta1ddotref, \
                theta2ref,theta2dotref,theta2ddotref, \
                parms.m1,parms.I1,parms.c1,parms.m2,parms.I2,parms.c2,parms.l,parms.g)
        z0 = np.array([z_temp[1,0]+np.random.normal(theta1_mean,theta1_dev), \
                    z_temp[1,1]+np.random.normal(theta1dot_mean,theta1dot_dev), \
                    z_temp[1,2]+np.random.normal(theta2_mean,theta2_dev), \
                    z_temp[1,3]+np.random.normal(theta2dot_mean,theta2dot_dev)])
        z[i+1] = z0
        T1[i+1,0] = T1_temp
        T2[i+1,0] = T2_temp

    animate(t,z,parms)
    plot(t,z,theta1_ref,theta2_ref)
