
import numpy as np

from matplotlib import pyplot as plt

def animate(t_interp, z_interp, params):

    lx, ly = params.lx, params.ly
    
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4
    ll = 1.5*(l1+l2)+0.2
    
    for i in range(0,len(t_interp)):
        theta1 = z_interp[i,0]
        theta2 = z_interp[i,2]
        theta3 = z_interp[i,4]
        theta4 = z_interp[i,6]

        O = np.array([0, 0])
        P1 = np.array([l1*np.sin(theta1), -l1*np.cos(theta1)])
        P2 = np.array([
            (l2*np.sin(theta1 + theta2)) + l1*np.sin(theta1),
            - (l2*np.cos(theta1 + theta2)) - l1*np.cos(theta1)
        ])
        
        O2 = np.array([lx, ly])
        P3 = np.array([
            lx + (l3*np.sin(theta3)),
            ly - (l3*np.cos(theta3))
        ])
        P4 = np.array([
            lx + (l4*np.sin(theta3 + theta4)) + l3*np.sin(theta3),
            ly - (l4*np.cos(theta3 + theta4)) - l3*np.cos(theta3)
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

def plot_result(t, z, t_ref, q1_refs, q2_refs, z_result, params):
    
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
    
    print(len(q1_refs), len(t_ref))
    
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(t_ref[1:],q1_refs[1:,0],'k--',color='red',label=r'$ \theta_{1ref} $')
    plt.plot(t_ref[1:],z_result[:,0],color='black',label=r'$ \theta_1 $')
    plt.plot(t_ref[1:],q2_refs[1:,0],'k--',color='blue',label=r'$ \theta_{2ref} $')
    if params.leg == "minitaur" or params.leg == "atrias":
        plt.plot(t_ref[1:],z_result[:,4],color='black',label=r'$ \theta_2 $')
    elif params.leg == "digit":
        plt.plot(t_ref[1:],z_result[:,6],color='black',label=r'$ \theta_2 $')
    plt.ylabel("angle reference")
    plt.legend(loc="upper left")
    
    plt.subplot(2, 1, 2)
    plt.plot(t_ref[1:],q1_refs[1:,1],'k--',color='red',label=r'$ w_{1ref}} $')
    plt.plot(t_ref[1:],z_result[:,1],color='black',label=r'$ w_1 $')
    plt.plot(t_ref[1:],q2_refs[1:,1],'k--',color='blue',label=r'$ w_{2ref}} $')
    if params.leg == "minitaur" or params.leg == "atrias":
        plt.plot(t_ref[1:],z_result[:,5],color='black',label=r'$ w_2 $')
    elif params.leg == "digit":
        plt.plot(t_ref[1:],z_result[:,7],color='black',label=r'$ w_2 $')
    plt.xlabel("t")
    plt.ylabel("velocity reference")
    plt.legend(loc="upper left")
    plt.show()