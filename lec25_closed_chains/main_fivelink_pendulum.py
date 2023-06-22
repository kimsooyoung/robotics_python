from matplotlib import pyplot as plt
import numpy as np

from scipy import interpolate
from scipy.integrate import odeint
from fivelinkpendulum_rhs import five_link_pendulum

class parameters:
    def __init__(self):
        
        self.m = 1
        self.I = 0.1
        self.l = 1
        self.g = 9.81
        
        self.pause = 0.02
        self.fps = 20
        
def cos(angle):
    return np.cos(angle)

def sin(angle):
    return np.sin(angle);

def interpolation(t, z, params):

    #interpolation
    t_interp = np.arange(t[0], t[len(t)-1], 1/params.fps)
    [rows, cols] = np.shape(z)
    z_interp = np.zeros((len(t_interp), cols))

    for i in range(0, cols):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    return t_interp, z_interp

def animate(t_interp, z_interp, params):

    l = params.l
    c1 = params.c1
    c2 = params.c2

    # #plot
    for i in range(0,len(t_interp)):
        theta1 = z_interp[i,0];
        theta2 = z_interp[i,2];
        O = np.array([0, 0])
        P = np.array([l*sin(theta1), -l*cos(theta1)])
        # 그림을 그려야 하니까 + P를 해주었음
        Q = P + np.array([l*sin(theta1+theta2),-l*cos(theta1+theta2)])
        
        # 사실 이렇게도 구할 수 있다.
        # H_01 = np.array([
        #     [np.cos(3*np.pi/2 + theta1), -np.sin(3*np.pi/2 + theta1), 0],
        #     [np.sin(3*np.pi/2 + theta1), -np.cos(3*np.pi/2 + theta1), 0],
        #     [0, 0, 1],
        # ])
        # H_12 = np.array([
        #     [np.cos(theta2), -np.sin(theta2), 0],
        #     [np.sin(theta2), -np.cos(theta2), 0],
        #     [0, 0, 1],
        # ])

        # COM Point
        G1 = np.array([c1*sin(theta1), -c1*cos(theta1)])
        G2 = P + np.array([c2*sin(theta1+theta2),-c2*cos(theta1+theta2)])

        pend1, = plt.plot([O[0], P[0]],[O[1], P[1]],linewidth=5, color='red')
        pend2, = plt.plot([P[0], Q[0]],[P[1], Q[1]],linewidth=5, color='blue')
        com1, = plt.plot(G1[0],G1[1],color='black',marker='o',markersize=10)
        com2, = plt.plot(G2[0],G2[1],color='black',marker='o',markersize=10)

        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.gca().set_aspect('equal')

        plt.pause(params.pause)

        if (i < len(t_interp)-1):
            pend1.remove()
            pend2.remove()
            com1.remove()
            com2.remove()

    #plt.show()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    # result plotting
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t,z[:,0],color='red',label='theta1');
    plt.plot(t,z[:,2],color='blue',label='theta2');
    plt.ylabel("angle")
    plt.legend(loc="upper left")
    
    plt.subplot(2, 1, 2)
    plt.plot(t,z[:,1],color='red',label='omega1');
    plt.plot(t,z[:,3],color='blue',label='omega2');
    plt.xlabel("t")
    plt.ylabel("angular rate")
    plt.legend(loc="lower left")

    plt.show()

if __name__=="__main__":

    params = parameters()

    total_time = 20
    t = np.linspace(0, total_time, 100*total_time)
    
    # initlal state
    # [theta1, omega1, ... theta5, omega5]
    z0 = np.array([
            np.pi/2, 0, 
            0, 0,
            0, 0,
            0, 0,
            0, 0,
        ])

    z = odeint(
        five_link_pendulum, z0, t, args=(params,),
        rtol=1e-12, atol=1e-12
    )
    # t_interp, z_interp = interpolation(t, z, params)


    # animate(t_interp, z_interp, params)