from matplotlib import pyplot as plt
import numpy as np
import control
import math
from scipy import interpolate
from scipy.integrate import odeint
import scipy.linalg #used by lqr

pi = np.pi

class parameters:
    def __init__(self):
        self.m1 = 1
        self.m2 = 1
        self.l = 1
        self.c1 = self.l/2
        self.c2 = self.l/2
        
        self.I1 = self.m1*self.l**2/12
        self.I2 = self.m2*self.l**2/12
        
        self.g = 9.81
        self.pause = 0.01
        self.fps =20
        # self.B = np.array([ [1], [0]]) #pendubot
        #self.B = np.array([ [0], [1]]) #acrobot
        self.B = np.array([ [1, 0], [0, 1]]) 

def cos(angle):
    return np.cos(angle)

def sin(angle):
    return np.sin(angle);

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

    #plot
    for i in range(0,len(t_interp)):
        theta1 = z_interp[i,0];
        theta2 = z_interp[i,1];
        O = np.array([0, 0])
        P = np.array([-l*sin(theta1), l*cos(theta1)])
        Q = P + np.array([-l*sin(theta1+theta2),l*cos(theta1+theta2)])
        G1 = np.array([-c1*sin(theta1), c1*cos(theta1)])
        G2 = P + np.array([-c2*sin(theta1+theta2),c2*cos(theta1+theta2)])

        pend1, = plt.plot([O[0], P[0]],[O[1], P[1]],linewidth=5, color='red')
        pend2, = plt.plot([P[0], Q[0]],[P[1], Q[1]],linewidth=5, color='blue')
        com1, = plt.plot(G1[0],G1[1],color='black',marker='o',markersize=10)
        com2, = plt.plot(G2[0],G2[1],color='black',marker='o',markersize=10)

        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.gca().set_aspect('equal')

        plt.pause(parms.pause)
        if (i < len(t_interp)-1):
            pend1.remove()
            pend2.remove()
            com1.remove()
            com2.remove()

    #plt.show()
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def controller (z,K):
    u = -K.dot(z)
    return u

def lqr(A,B,Q,R):
    """Solve the continuous time lqr controllerler.

    Source: http://www.mwm.im/lqr-controllerlers-with-python/

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))


    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(np.matrix(B.T)*np.matrix(X)))

    eigVals, eigVecs = scipy.linalg.eig(A-B*K)

    return K, X, eigVals

def linearization(z,m1,m2,I1,I2,c1,c2,l,g,B):

    theta1 = z[0];
    theta2 = z[1];
    omega1 = z[2];
    omega2 = z[3];

    M11 =  1.0*I1 + 1.0*I2 + c1**2*m1 + m2*(c2**2 + 2*c2*l*cos(theta2) + l**2)
    M12 =  1.0*I2 + c2*m2*(c2 + l*cos(theta2))
    M21 =  1.0*I2 + c2*m2*(c2 + l*cos(theta2))
    M22 =  1.0*I2 + c2**2*m2
    M = np.array([[M11, M12], [M21,M22]]);

    dGdq11 =  -g*(c1*m1*cos(theta1) + c2*m2*cos(theta1 + theta2) + l*m2*cos(theta1))
    dGdq12 =  -c2*g*m2*cos(theta1 + theta2)
    dGdq21 =  -c2*g*m2*cos(theta1 + theta2)
    dGdq22 =  -c2*g*m2*cos(theta1 + theta2)
    dGdq = np.array([[dGdq11, dGdq12], [dGdq21,dGdq22]]);

    dGdqdot11 =  0
    dGdqdot12 =  0
    dGdqdot21 =  0
    dGdqdot22 =  0
    dGdqdot = np.array([[dGdqdot11, dGdqdot12], [dGdqdot21,dGdqdot22]]);

    dCdq11 =  0
    dCdq12 =  -c2*l*m2*omega2*(2.0*omega1 + 1.0*omega2)*cos(theta2)
    dCdq21 =  0
    dCdq22 =  1.0*c2*l*m2*omega1**2*cos(theta2)
    dCdq = np.array([[dCdq11, dCdq12], [dCdq21,dCdq22]]);

    dCdqdot11 =  -2.0*c2*l*m2*omega2*sin(theta2)
    dCdqdot12 =  -2.0*c2*l*m2*(omega1 + omega2)*sin(theta2)
    dCdqdot21 =  2*c2*l*m2*omega1*sin(theta2)
    dCdqdot22 =  0
    dCdqdot = np.array([[dCdqdot11, dCdqdot12], [dCdqdot21,dCdqdot22]]);

    invM = np.linalg.inv(M)

    dGCdq = dGdq + dCdq
    dGCdqdot = dGdqdot + dCdqdot

    # A_lin = [0(2x2) I(2x2);
    #      invM*(dGC/dq) invM*(dGC/dqdot)]
    A_lin = np.block([
        [np.zeros((2, 2)), np.identity(2)],
        [np.matmul(invM,-dGCdq), np.matmul(invM,-dGCdqdot)]
     ])

    # B_lin = [0(2x2); invM*B]
    B_lin = np.block([
        [np.zeros((2, 2))],
        [np.matmul(invM,B)]
    ])

    return A_lin, B_lin

def plot(t,z,u):

    plt.figure(1)
    
    plt.subplot(4, 1, 1)
    plt.plot(t,z[:,0],color='red',label='theta1');
    plt.plot(t,z[:,1],color='blue',label='theta2');
    plt.ylabel("angle")
    plt.legend(loc="upper left")
    
    plt.subplot(4, 1, 2)
    plt.plot(t,z[:,2],color='red',label='omega1');
    plt.plot(t,z[:,3],color='blue',label='omega2');
    plt.xlabel("t")
    plt.ylabel("angular rate")
    plt.legend(loc="lower left")
    
    plt.subplot(4, 1, 3)
    plt.plot(t,u[:,0],color='green');
    plt.xlabel("t")
    plt.ylabel("torque1")
    
    plt.subplot(4, 1, 4)
    plt.plot(t,u[:,1],color='green');
    plt.xlabel("t")
    plt.ylabel("torque2")
        
    plt.show()
    
def double_pendulum(z,t,m1,m2,I1,I2,c1,c2,l,g,B,K,T1_disturb,T2_disturb):

    theta1 = z[0];
    theta2 = z[1];
    omega1 = z[2];
    omega2 = z[3];
    #theta1,theta2,omega1,omega2 = z

    M11 =  1.0*I1 + 1.0*I2 + c1**2*m1 + m2*(c2**2 + 2*c2*l*cos(theta2) + l**2)
    M12 =  1.0*I2 + c2*m2*(c2 + l*cos(theta2))
    M21 =  1.0*I2 + c2*m2*(c2 + l*cos(theta2))
    M22 =  1.0*I2 + c2**2*m2

    C1 =  -c2*l*m2*omega2*(2.0*omega1 + 1.0*omega2)*sin(theta2)
    C2 =  c2*l*m2*omega1**2*sin(theta2)

    G1 =  -g*(c1*m1*sin(theta1) + c2*m2*sin(theta1 + theta2) + l*m2*sin(theta1))
    G2 =  -c2*g*m2*sin(theta1 + theta2)

    u  = controller(z,K)
    Bu = (B@u).reshape(2,1)
    T_disturb = np.array([[T1_disturb],[T2_disturb]])
    # print(np.shape(Bu))


    M = np.array([[M11, M12], [M21,M22]]);
    CG = np.array([[C1+G1],[C2+G2]])
    invM = np.linalg.inv(M)
    thetaddot = np.matmul(invM,-CG+Bu+T_disturb) #invM.dot(-CG)
    # print(thetaddot)
    alpha1, alpha2 = thetaddot[0,0], thetaddot[1,0]

    dzdt = np.array([omega1, omega2, alpha1, alpha2]);
    return dzdt

if __name__=="__main__":
    #parameters
    parms = parameters()

    # disturbances
    T1_mean, T1_dev = 0, 0.0 * 1
    T2_mean, T2_dev = 0, 0.0 * 1
    theta1_mean, theta1_dev = 0, 0.0 * 1
    theta2_mean, theta2_dev = 0, 0.0 * 1
    theta1dot_mean, theta1dot_dev = 0, 0.0 * 1
    theta2dot_mean, theta2dot_dev = 0, 0.0 * 1
    
    #compute controller K
    #linearize about vertical position
    theta1, theta2, omega1, omega2 = 0, 0, 0, 0
    z = np.array([theta1,theta2,omega1,omega2])

    A_lin,B_lin = linearization(z,parms.m1,parms.m2,parms.I1,parms.I2,\
            parms.c1,parms.c2,parms.l,parms.g,parms.B)
    Q = np.eye((4))
    R = 1e-2 * np.eye((2))
    # K, X, eigVals = lqr(A_lin,B_lin,Q,R) #hand coded
    K, S, E = control.lqr(A_lin,B_lin,Q,R) #from python module lqr
    print("K = ", K)
    print(f"E : {E}")

    N = 101
    t0 = 0; tf = 10
    t = np.linspace(t0, tf, N)
    z0 = np.array([pi/4, -pi/4, 0, 0])
    shape = (N,4) #2 is for theta1 and theta2 and their rates, change according to the system
    
    z = np.zeros(shape)
    u = np.zeros((N,2))
    z[0] = z0

    for i in range(0,N-1):
        T1_disturb = np.random.normal(T1_mean,T1_dev)
        T2_disturb = np.random.normal(T2_mean,T2_dev)
        physical_parms = (parms.m1,parms.m2,parms.I1,parms.I2,parms.c1,parms.c2, parms.l,parms.g,parms.B)
        control_parms = (K,T1_disturb,T2_disturb)
        all_parms = physical_parms + control_parms
        t_temp = np.array([t[i], t[i+1]])
        z_temp = odeint(double_pendulum, z0, t_temp, args=all_parms)
        u_temp = controller(z0,K)
        z0 = np.array([z_temp[1,0]+np.random.normal(theta1_mean,theta1_dev), \
                    z_temp[1,1]+np.random.normal(theta1dot_mean,theta1dot_dev), \
                    z_temp[1,2]+np.random.normal(theta2_mean,theta2_dev), \
                    z_temp[1,3]+np.random.normal(theta2dot_mean,theta2dot_dev)])
        z[i+1] = z0
        u[i+1] = u_temp

    animate(t,z,parms)
    plot(t,z,u)

