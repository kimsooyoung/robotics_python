from scipy.integrate import odeint
from scipy import interpolate

import matplotlib.pyplot as plt
import numpy as np

class Parameters:
    def __init__(self):
        self.m1 = 1
        self.g = 9.81
        self.l1 = 1
        self.I1 = (self.m1 / 12) * (self.l1 ** 2)
        self.pause = 0.01
        self.fps = 20

        self.kp1 = 200
        self.kd1 = 2*np.sqrt(self.kp1)

def get_dynamics(theta, m, I, g, l):
    
    M = m*l**2/4 + I
    C = 0
    G = m*g*l/2*np.cos(theta)
    
    return M, C, G

def get_tau(theta, omega, m, I, g, l, q_ref, qd_ref, qdd_ref, kp, kd):

    M, C, G = get_dynamics(theta, m, I, g, l)
    
    return M*(qdd_ref - kp*(theta-q_ref) - kd*(omega-qd_ref)) + C + G

def one_link_eom(z,t, m, I, g, l, kp, kd, theta_ref, omega_ref, ang_acc_ref):
    
    theta, omega = z
    
    M, C, G = get_dynamics(theta, m, I, g, l)
    
    tau = get_tau(theta,omega,m,I,g,l,theta_ref,omega_ref,ang_acc_ref,kp,kd)
    ang_acc = (tau - C - G)/M
    
    return np.array([omega, ang_acc])
        

def animate(t,z,parms):
    #interpolation
    t_interp = np.arange(t[0],t[len(t)-1],1/parms.fps)
    m, n = np.shape(z)
    z_interp = np.zeros((len(t_interp),n))

    for i in range(0,n-1):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    l1 = parms.l1

    #plot
    for i in range(0,len(t_interp)):
        theta1 = z_interp[i,0];
        O = np.array([0, 0])
        P = np.array([l1*np.cos(theta1), l1*np.sin(theta1)])

        pend1, = plt.plot([O[0], P[0]],[O[1], P[1]],linewidth=5, color='red')

        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.gca().set_aspect('equal')

        plt.pause(parms.pause)
        pend1.remove()

    plt.close()        
        

def plot(t, z, theta_ref, omega_ref, T):

    plt.figure(1)

    plt.subplot(3,1,1)
    plt.plot(t,z[:,0])
    plt.plot(t,theta_ref,'r');
    plt.ylabel("theta1")
    plt.title("Plot of position, velocity, and Torque vs. time")
    
    plt.subplot(3,1,2)
    plt.plot(t,z[:,1])
    plt.plot(t,omega_ref,'r');
    plt.ylabel("theta1dot")
    
    plt.subplot(3,1,3)
    plt.plot(t,T[:,0])
    plt.xlabel("t")
    plt.ylabel("Torque")

    plt.show()        
        
if __name__=="__main__":
    
    params = Parameters()
    m,I,g,l,kp,kd = params.m1, params.I1, params.g, params.l1, params.kp1, params.kd1
    
    t0, t1, t2 = 0, 1.5, 3
    
    ts1 = np.linspace(t0, t1, 100)
    ts2 = np.linspace(t1, t2, 100)
    
    ts = np.concatenate((ts1, ts2[1:]))
    
    pi = np.pi
    a10 = 0
    a11 = 0
    a12 = 0.666666666666667*pi
    a13 = -0.296296296296296*pi
    a20 = -2.0*pi
    a21 = 4.0*pi
    a22 = -2.0*pi
    a23 = 0.296296296296296*pi
    
    theta1 = a10 + a11*ts1 + a12*ts1**2 + a13*ts1**3
    omega1 = a11 + 2*a12*ts1 + 3*a13*ts1**2
    ang_acc1 = 2*a12 + 6*a13*ts1
    
    theta2 = a20 + a21*ts2 + a22*ts2**2 + a23*ts2**3
    omega2 = a21 + 2*a22*ts2 + 3*a23*ts2**2
    ang_acc2 = 2*a22 + 6*a23*ts2
    
    theta_ref = np.concatenate((theta1, theta2[1:]))
    omega_ref = np.concatenate((omega1, omega2[1:]))
    ang_acc_ref = np.concatenate((ang_acc1, ang_acc2[1:]))
    
    # initial conditions
    z0, tau0 = np.array([0, 0]), np.array([0])
    
    z = np.zeros((len(ts), 2))
    tau = np.zeros((len(ts), 1))
    z[0], tau[0] = z0, tau0
    
    for i in range(len(ts)-1):
        
        args = m,I,g,l,kp,kd, theta_ref[i],omega_ref[i],ang_acc_ref[i]
        
        temp_ts = np.array([ts[i], ts[i+1]])
        result = odeint(one_link_eom, z0, temp_ts, args=args)
        temp_tau = get_tau(z0[0], z0[1], m, I, g, l, theta_ref[i], omega_ref[i], ang_acc_ref[i], kp, kd)
        
        # 만약 실제 로봇이었다면, 여기가 sensor로부터 받은 값이 될 것이다.
        z0 = result[1]
        
        z[i+1] = z0
        tau[i+1] = temp_tau
        
    animate(ts, z, params)
    plot(ts, z, theta_ref, omega_ref, tau)