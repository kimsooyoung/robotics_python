from matplotlib import pyplot as plt
import numpy as np

from scipy import interpolate
from scipy.optimize import fsolve
from scipy.integrate import odeint

from fourlinkchain_rhs import fourlinkchain

class parameters:
    def __init__(self):
        
        self.m1 = 1; self.m2 = 1; self.m3 = 1; self.m4 = 1;
        self.I1 = 0.1; self.I2 = 0.1; self.I3 = 0.1; self.I4 = 0.1;
        
        # ### minitaur leg length ###
        # self.l1 = 1; self.l2 = 2; self.l3 = 1; self.l4 = 2;
        
        ### atrias/digit leg ###
        self.l1 = 1; self.l2 = 2; self.l3 = 2; self.l4 = 1;
        
        self.lx = 0; self.ly = 0;
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
        
        plt.xlim([-ll, ll])
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


def plot_result(t, z):
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t,z[:,0],color='red',label=r'$ \theta_1 $');
    plt.plot(t,z[:,2],color='green',label=r'$ \theta_2 $');
    plt.plot(t,z[:,4],color='blue',label=r'$ \theta_3 $');
    plt.plot(t,z[:,6],color='cyan',label=r'$ \theta_4 $');
    plt.ylabel("angle")
    plt.legend(loc="upper left")
    
    plt.subplot(2, 1, 2)
    plt.plot(t,z[:,1],color='red',label=r'$ w_1 $');
    plt.plot(t,z[:,3],color='green',label=r'$ w_2 $');
    plt.plot(t,z[:,5],color='blue',label=r'$ w_3 $');
    plt.plot(t,z[:,7],color='cyan',label=r'$ w_4 $');
    plt.xlabel("t")
    plt.ylabel("angular rate")
    plt.legend(loc="lower left")

    plt.show()

def position_last_link_tip(z, params):
    
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4
    lx, ly = params.lx, params.ly
    
    q1, q2, q3, q4 = z
    
    del_x = l2*sin(q1 + q2) - lx - l4*sin(q3 + q4) + l1*sin(q1) - l3*sin(q3)
    del_y = l4*cos(q3 + q4) - l2*cos(q1 + q2) - ly - l1*cos(q1) + l3*cos(q3)
    
    return del_x, del_y, 0, 0

def velocity_last_link_tip(z, params, q_star):
    
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4
    q1, q2, q3, q4 = q_star
    u1, u2, u3, u4 = z
    
    del_vx = u1*(l2*cos(q1 + q2) + l1*cos(q1)) - u3*(l4*cos(q3 + q4) + l3*cos(q3)) + l2*u2*cos(q1 + q2) - l4*u4*cos(q3 + q4);
    del_vy = u1*(l2*sin(q1 + q2) + l1*sin(q1)) - u3*(l4*sin(q3 + q4) + l3*sin(q3)) + l2*u2*sin(q1 + q2) - l4*u4*sin(q3 + q4);

    return del_vx, del_vy, 0, 0

if __name__=="__main__":

    params = parameters()

    z = None
    total_time = 5
    t = np.linspace(0, total_time, 100*total_time)
    
    ### Solve q's such that end of final link is at lx,ly ###
    q1, q2, q3, q4 = -np.pi/3, np.pi/2, np.pi/3, -np.pi/2 
    q0 = [q1, q2, q3, q4]
    q_star = fsolve(position_last_link_tip, q0, params)
    q1, q2, q3, q4 = q_star
    print(f"q1: {q1}, q2: {q2}, q3: {q3}, q4: {q4}")
    
    ### Solve u's such that end of final link is linear velocity 0,0 ###
    u1, u2, u3, u4 = 0, 0, 0, 0
    u0 = [u1, u2, u3, u4]
    fsolve_params = (params, q_star)
    u_star = fsolve(velocity_last_link_tip, u0, fsolve_params)
    u1, u2, u3, u4 = u_star
    print(f"u1: {u1}, u2: {u2}, u3: {u3}, u4: {u4}")
    
    ### Use ode45 to do simulation ###
    z0 = np.array([
        q1, u1,
        q2, u2,
        q3, u3,
        q4, u4
    ])

    try:
        z = odeint(
            fourlinkchain, z0, t, args=(params,),
            rtol=1e-9, atol=1e-9, mxstep=5000
        )
    except Exception as e:
        print(e)
    finally:
        t_interp, z_interp = interpolation(t, z, params)
        animate(t_interp, z_interp, params)
        plot_result(t, z)
        print("done")
