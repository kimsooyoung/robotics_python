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
        self.g = 10
        self.l1 = 1
        self.I1 = 1/12 * (self.m1 * self.l1**2)
        
        self.kp1 = 200
        self.kd1 = 2 * np.sqrt(self.kp1)
        
        self.theta_des = np.pi/2
        
        self.pause = 0.001
        self.fps = 30
        
def animate(t,z,parms):
    #interpolation
    t_interp = np.arange(t[0],t[len(t)-1],1/parms.fps)
    # N 2
    m, n = np.shape(z)
    shape = (len(t_interp),n)
    z_interp = np.zeros(shape)

    for i in range(n):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    l1 = parms.l1

    plt.figure(1)
    
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.gca().set_aspect('equal')

    #plot
    for i in range(len(t_interp)):
        theta1 = z_interp[i,0];
        O = np.array([0, 0])
        P = np.array([l1*cos(theta1), l1*sin(theta1)])

        pend1, = plt.plot([O[0], P[0]],[O[1], P[1]],linewidth=5, color='red')
        
        plt.pause(parms.pause)
        pend1.remove()

    plt.close()

def control(theta1,theta1ref,theta1dot,kp1,kd1):
    T1 = -kp1*(theta1-theta1ref)-kd1*theta1dot
    return T1

def onelink_rhs(z,t,m1,I1,l1,g,kp1,kd1,theta1ref,T1_disturb):

    theta1 = z[0]
    theta1dot = z[1]

    T1 = control(theta1,theta1ref,theta1dot,kp1,kd1)-T1_disturb
    theta1ddot = (1/( I1+(m1*l1*l1/4) ))*(T1 - 0.5*m1*g*l1*cos(theta1));

    zdot = np.array([theta1dot, theta1ddot]);

    return zdot

def plot(t,z,T,parms):

    plt.figure(2)

    plt.subplot(3,1,1)
    plt.plot(t, parms.theta_des * np.ones(len(t)), 'r-.');
    plt.plot(t,z[:,0])
    plt.ylabel("theta1")
    plt.title("Plot of position, velocity, and Torque vs. time")

    plt.subplot(3,1,2)
    plt.plot(t, z[:,1])
    plt.ylabel("theta1dot")

    plt.subplot(3,1,3)
    plt.plot(t, T[:,0])
    plt.xlabel("t")
    plt.ylabel("Torque")

    plt.show()

if __name__=="__main__":

    #parameters
    parms = parameters()

    #initialization
    theta1, theta1dot = 0, 0;

    # disturbances
    T1_mean, T1_dev = 0, 40 * 0.1
    theta1_mean, theta1_dev = 0, 0.0    
    theta1dot_mean, theta1dot_dev = 0, 0.5 * 0

    #time
    t0, tend = 0, 2

    # h = 0.005;
    # N = int((tend-t0)/h) + 1;
    N = 200
    t = np.linspace(t0, tend, N)

    #state
    # 2 is for theta1 and theta1dot, change according to the system
    z = np.zeros((N, 2))
    tau = np.zeros((N, 1))
    
    z0 = np.array([theta1, theta1dot])
    z[0] = z0
    
    physical_parms = (parms.m1, parms.I1, parms.l1, parms.g)
    control_parms = (parms.kp1, parms.kd1, parms.theta_des)
    
    for i in range(len(t)-1):
        T1_disturb = (np.random.normal(T1_mean, T1_dev),)
        all_parms = physical_parms + control_parms + T1_disturb
        
        t_temp = np.array([t[i], t[i+1]])
        z_temp = odeint(onelink_rhs, z0, t_temp, args=all_parms)
        
        # 실제 노이즈는 odeint단에서 추가된다.
        # 우리가 로봇을 실제 움직일 때도 
        # 토크는 제대로 주는데 로봇이 그대로 안움직이니까
        tau_temp = control(
            z0[0], parms.theta_des, 
            z0[1], parms.kp1, parms.kd1
        )
        
        z0 = np.array([
            z_temp[1,0]+ np.random.normal(theta1_mean, theta1_dev),
            z_temp[1,1]+ np.random.normal(theta1dot_mean, theta1dot_dev)
        ])
        
        z[i+1] = z0
        tau[i+1,0] = tau_temp
        
    animate(t, z, parms)
    plot(t, z, tau, parms)