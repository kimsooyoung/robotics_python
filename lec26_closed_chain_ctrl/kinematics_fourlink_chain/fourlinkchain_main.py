from matplotlib import pyplot as plt
import numpy as np

from scipy import interpolate
from scipy.optimize import fsolve
from scipy.integrate import odeint

from quinticpolytraj import quinticpolytraj

class parameters:
    def __init__(self):

        # ### minitaur leg length ###
        # self.l1 = 1; self.l2 = 2; self.l3 = 1; self.l4 = 2;
        
        ### atrias/digit leg ###
        self.l1 = 1; self.l2 = 2; self.l3 = 2; self.l4 = 1;
        
        self.lx = 0; self.ly = 0;
        self.g = 9.81
        
        self.show_phase = False
        self.pause = 0.01
        self.fps = 20
        
def cos(angle):
    return np.cos(angle)

def sin(angle):
    return np.sin(angle);

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

    plt.show()

def plot_traj(t, q_ref, q1d_ref, q3d_ref, q1dd_ref, q3dd_ref):
    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(t,q_ref[:,0],color='red',label=r'$ \theta_1 $');
    plt.plot(t,q_ref[:,2],color='green',label=r'$ \theta_2 $');
    plt.plot(t,q_ref[:,4],color='blue',label=r'$ \theta_3 $');
    plt.plot(t,q_ref[:,6],color='cyan',label=r'$ \theta_4 $');
    plt.ylabel("angle")
    plt.legend(loc="upper left")
    
    plt.subplot(3, 1, 2)
    plt.plot(t,q1d_ref,'k--',color='red',label=r'$ w_1 $');
    plt.plot(t,q3d_ref,'k--',color='blue',label=r'$ w_3 $');
    plt.xlabel("t")
    plt.ylabel("angular velocity")
    plt.legend(loc="upper left")

    plt.subplot(3, 1, 3)
    plt.plot(t,q1dd_ref,'k-.',color='red',label=r'$ a_1 $');
    plt.plot(t,q3dd_ref,'k-.',color='blue',label=r'$ a_3 $');
    plt.xlabel("t")
    plt.ylabel("angular acceleration")
    plt.legend(loc="upper left")

    plt.show()

def position_kinematics(z, params, l, alpha):
    
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4
    lx, ly = params.lx, params.ly
    
    q1, q2, q3, q4 = z
    
    del_x = l1*sin(q1) + l2*sin(q1 + q2) - l3*sin(q3) - l4*sin(q3 + q4) - lx
    del_y = -l1*cos(q1) - l2*cos(q1 + q2) + l3*cos(q3) + l4*cos(q3 + q4) - ly
    
    leg_length = np.sqrt(l1**2 + 2*l1*l2*cos(q2) + l2**2)
    leg_angle = 0.5*q1 + 0.5*q3
    
    return del_x, del_y, leg_length - l, leg_angle - alpha

def position_last_link_tip(z, params, q1, q3):
    
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4
    lx, ly = params.lx, params.ly
    
    q2, q4 = z
    
    del_x = l2*sin(q1 + q2) - lx - l4*sin(q3 + q4) + l1*sin(q1) - l3*sin(q3)
    del_y = l4*cos(q3 + q4) - l2*cos(q1 + q2) - ly - l1*cos(q1) + l3*cos(q3)
    
    return del_x, del_y

if __name__=="__main__":

    params = parameters()
    show_phase = params.show_phase
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4

    z = None
    total_time = 5
    t = np.linspace(0, total_time, 100*total_time)
    
    ### Solve q's such that end of final link is at lx,ly ###
    q1, q2, q3, q4 = -np.pi/3, np.pi/2, 0, 0
    
    ### kinematics initial condition ###
    l = 0.9 * (l1 + l2)
    alpha = -0.5
    q0 = [q1, q2, q3, q4]
    fsolve_params = (params, l, alpha)
    q_ini = fsolve(position_kinematics, q0, fsolve_params)
    q1, q2, q3, q4 = q_ini
    print(f"[initial condition] q1: {q1}, q2: {q2}, q3: {q3}, q4: {q4}")
    
    ### kinematics middle condition ###
    l = 0.75 * (l1 + l2)
    alpha = 0.0
    q0 = [q1, q2, q3, q4]
    fsolve_params = (params, l, alpha)
    q_middle = fsolve(position_kinematics, q0, fsolve_params)
    q1, q2, q3, q4 = q_middle
    print(f"[middle condition] q1: {q1}, q2: {q2}, q3: {q3}, q4: {q4}")
    
    ### kinematics final condition ###
    l = 0.9 * (l1 + l2)
    alpha = 0.5
    q0 = [q1, q2, q3, q4]
    fsolve_params = (params, l, alpha)
    q_final = fsolve(position_kinematics, q0, fsolve_params)
    q1, q2, q3, q4 = q_final
    print(f"[final condition] q1: {q1}, q2: {q2}, q3: {q3}, q4: {q4}")
    
    if show_phase:
        z_ini = [ q_ini[0], 0, q_ini[1], 0, q_ini[2], 0, q_ini[3], 0 ]
        animate([0], np.reshape(z_ini, (1,8)), params)
    
        z_middle = [ q_middle[0], 0, q_middle[1], 0, q_middle[2], 0, q_middle[3], 0 ]
        animate([0], np.reshape(z_middle, (1,8)), params)
    
        z_final = [ q_final[0], 0, q_final[1], 0, q_final[2], 0, q_final[3], 0 ]
        animate([0], np.reshape(z_final, (1,8)), params)
    
    ### Traj generation ###
    t_init, t_mid, t_end = 0, 1.0, 2.0
    
    q1_ref, q1d_ref, q1dd_ref, t_ref = quinticpolytraj(q_ini[0], q_middle[0], q_final[0], t_init, t_mid, t_end)
    q3_ref, q3d_ref, q3dd_ref, t_ref = quinticpolytraj(q_ini[2], q_middle[2], q_final[2], t_init, t_mid, t_end)
    
    ### Calc q2_ref, q4_ref ###
    q_ref = np.zeros( (len(t_ref), 8) )
    q2_ref, q4_ref = [], []
    z0 = [0.5, -0.5]
    for i in range(len(t_ref)):
        fsolve_params = (params, q1_ref[i], q3_ref[i])
        q_sol = fsolve(position_last_link_tip, z0, fsolve_params)
        q2_ref.append(q_sol[0])
        q4_ref.append(q_sol[1])
        
        # Aggregate All q refs
        q_ref[i, 0] = q1_ref[i]; q_ref[i, 1] = 0
        q_ref[i, 2] = q_sol[0];  q_ref[i, 3] = 0
        q_ref[i, 4] = q3_ref[i]; q_ref[i, 5] = 0
        q_ref[i, 6] = q_sol[1];  q_ref[i, 7] = 0
        
        # prevent error case
        z0 = [ q_sol[0], q_sol[1] ]
        
    animate(t_ref, q_ref, params)
    plot_traj(t_ref, q_ref, q1d_ref, q3d_ref, q1dd_ref, q3dd_ref)
    
