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
    # [rows, cols] = np.shape(z)
    [cols, rows] = np.shape(z)
    z_interp = np.zeros((len(t_interp), rows))

    for i in range(0, rows):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    return t_interp, z_interp

def animate(t_interp, z_interp, params):

    l = params.l
    ll = 5*l+0.2

    # #plot
    for i in range(0,len(t_interp)):
        theta1 = z_interp[i,0]
        theta2 = z_interp[i,2]
        theta3 = z_interp[i,4]
        theta4 = z_interp[i,6]
        theta5 = z_interp[i,8]

        O = np.array([0, 0])
        P1 = np.array([l*sin(theta1), -l*cos(theta1)])
        P2 = P1 + l*np.array([sin(theta1+theta2), -cos(theta1+theta2)])
        P3 = P2 + l*np.array([sin(theta1+theta2+theta3), -cos(theta1+theta2+theta3)])
        P4 = P3 + l*np.array([sin(theta1+theta2+theta3+theta4), -cos(theta1+theta2+theta3+theta4)])
        P5 = P4 + l*np.array([sin(theta1+theta2+theta3+theta4+theta5), -cos(theta1+theta2+theta3+theta4+theta5)])
        
        h1, = plt.plot([O[0], P1[0]],[O[1], P1[1]],linewidth=5, color='red')
        h2, = plt.plot([P1[0], P2[0]],[P1[1], P2[1]],linewidth=5, color='green')
        h3, = plt.plot([P2[0], P3[0]],[P2[1], P3[1]],linewidth=5, color='blue')
        h4, = plt.plot([P3[0], P4[0]],[P3[1], P4[1]],linewidth=5, color='cyan')
        h5, = plt.plot([P4[0], P5[0]],[P4[1], P5[1]],linewidth=5, color='magenta')
        
        # pend1, = plt.plot([O[0], P[0]],[O[1], P[1]],linewidth=5, color='red')
        # pend2, = plt.plot([P[0], Q[0]],[P[1], Q[1]],linewidth=5, color='blue')
        
        # # COM Point
        # G1 = np.array([c1*sin(theta1), -c1*cos(theta1)])
        # G2 = P + np.array([c2*sin(theta1+theta2),-c2*cos(theta1+theta2)])

        # com1, = plt.plot(G1[0],G1[1],color='black',marker='o',markersize=10)
        # com2, = plt.plot(G2[0],G2[1],color='black',marker='o',markersize=10)

        plt.xlim(-ll, ll)
        plt.ylim(-ll ,ll)
        plt.gca().set_aspect('equal')

        plt.pause(params.pause)

        if (i < len(t_interp)-1):
            h1.remove()
            h2.remove()
            h3.remove()
            h4.remove()
            h5.remove()
            # com1.remove()
            # com2.remove()

    #plt.show()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    # # result plotting
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t,z[:,0],color='red',label='theta1');
    plt.plot(t,z[:,2],color='green',label='theta2');
    plt.plot(t,z[:,4],color='blue',label='theta2');
    plt.plot(t,z[:,6],color='cyan',label='theta2');
    plt.plot(t,z[:,8],color='magenta',label='theta2');
    plt.ylabel("angle")
    plt.legend(loc="upper left")
    
    plt.subplot(2, 1, 2)
    plt.plot(t,z[:,1],color='red',label='omega1');
    plt.plot(t,z[:,3],color='green',label='omega2');
    plt.plot(t,z[:,5],color='blue',label='omega2');
    plt.plot(t,z[:,7],color='cyan',label='omega2');
    plt.plot(t,z[:,9],color='magenta',label='omega2');
    plt.xlabel("t")
    plt.ylabel("angular rate")
    plt.legend(loc="lower left")

    plt.show()


def test(z0, t, params):

    q1, u1, q2, u2, q3, u3, q4, u4, q5, u5 = z0

    return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

if __name__=="__main__":

    params = parameters()

    z = None
    total_time = 2
    t = np.linspace(0.0, total_time, 100*total_time)
    
    # initlal state
    # [theta1, omega1, ... theta5, omega5]
    # z0 = np.array([ np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z0 = np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    try:
        z = odeint(
            test, z0, t, args=(params,),
            rtol=1e-12, atol=1e-12
        )
    except Exception as e:
        print(e)
    finally:
        # t_interp, z_interp = interpolation(t, z, params)
        # animate(t_interp, z_interp, params)
        print("done")