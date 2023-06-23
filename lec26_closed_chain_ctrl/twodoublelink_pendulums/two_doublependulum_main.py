from matplotlib import pyplot as plt
import numpy as np

from scipy import interpolate
from scipy.optimize import fsolve
from scipy.integrate import odeint

from two_doublependulum_rhs import two_double_pendulum

class parameters:
    def __init__(self):
        
        self.m1 = 1; self.m2 = 1; self.m3 = 1; self.m4 = 1;
        self.l1 = 1; self.l2 = 1; self.l3 = 1; self.l4 = 1;
        self.I1 = 0.1; self.I2 = 0.1; self.I3 = 0.1; self.I4 = 0.1;
        
        self.lx = 2; self.ly = 0;
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

    lx, ly = params.lx, params.ly
    
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4
    ll = 1.5*(l1+l2)+0.2
    
    # #plot
    for i in range(0,len(t_interp)):
        theta1 = z_interp[i,0]
        theta2 = z_interp[i,2]
        theta3 = z_interp[i,4]
        theta4 = z_interp[i,6]

        O = np.array([0, 0])
        P1 = np.array([l1*sin(theta1), -l1*cos(theta1)])
        P2 = np.array([
            (l2*sin(theta1 + theta2)) + l1*sin(theta1),
            - (l2*cos(theta1 + theta2)) - l1*cos(theta1)
        ])
        
        O2 = np.array([lx, ly])
        P3 = np.array([
            lx + (l3*sin(theta3)),
            ly - (l3*cos(theta3))
        ])
        P4 = np.array([
            lx + (l4*sin(theta3 + theta4)) + l3*sin(theta3),
            ly - (l4*cos(theta3 + theta4)) - l3*cos(theta3)
        ])
        
        h1, = plt.plot([O[0], P1[0]],[O[1], P1[1]],linewidth=5, color='red')
        h2, = plt.plot([P1[0], P2[0]],[P1[1], P2[1]],linewidth=5, color='green')
        h3, = plt.plot([O2[0], P3[0]],[O2[1], P3[1]],linewidth=5, color='blue')
        h4, = plt.plot([P3[0], P4[0]],[P3[1], P4[1]],linewidth=5, color='cyan')
        
        plt.xlim([-ll+0.25*ll, ll+0.25*ll])
        plt.ylim([-ll, ll])
        plt.gca().set_aspect('equal')

        plt.pause(params.pause)

        if (i < len(t_interp)-1):
            h1.remove()
            h2.remove()
            h3.remove()
            h4.remove()

    #plt.show()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

if __name__=="__main__":

    params = parameters()

    z = None
    total_time = 5
    t = np.linspace(0, total_time, 100*total_time)
    
    ### Use ode45 to do simulation ###
    z0 = np.array([
        np.pi/2 + 0.01, 0,
        0, 0,
        np.pi/2, 0,
        0, 0
    ])

    try:
        z = odeint(
            two_double_pendulum, z0, t, args=(params,),
            rtol=1e-12, atol=1e-12
        )
    except Exception as e:
        print(e)
    finally:
        t_interp, z_interp = interpolation(t, z, params)
        animate(t_interp, z_interp, params)
        print("done")
