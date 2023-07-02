import numpy as np 
import matplotlib.pyplot as plt

def cos(angle): 
    return np.cos(angle) 

def sin(angle): 
    return np.sin(angle) 

def nlink_animate(t_interp, z_interp, params): 

    l_0 = params.l1 
    l_1 = params.l2 
    l_2 = params.l3 

    ll = params.l1*4 + 0.2
    for i in range(0,len(t_interp)):
        q_0 = z_interp[i,0]
        q_1 = z_interp[i,2]
        q_2 = z_interp[i,4]

        P0 = np.array([0, 0])
        P1 = np.array([l_0*sin(q_0),-l_0*cos(q_0)])
        P2 = np.array([l_0*sin(q_0) + l_1*sin(q_0 + q_1),-l_0*cos(q_0) - l_1*cos(q_0 + q_1)])
        P3 = np.array([l_0*sin(q_0) + l_1*sin(q_0 + q_1) + l_2*sin(q_0 + q_1 + q_2),-l_0*cos(q_0) - l_1*cos(q_0 + q_1) - l_2*cos(q_0 + q_1 + q_2)])

        h1, = plt.plot([P0[0], P1[0]],[P0[1], P1[1]],linewidth=5, color='yellow')
        h2, = plt.plot([P1[0], P2[0]],[P1[1], P2[1]],linewidth=5, color='red')
        h3, = plt.plot([P2[0], P3[0]],[P2[1], P3[1]],linewidth=5, color='green')

        plt.xlim([-ll, ll])
        plt.ylim([-ll, ll])
        plt.gca().set_aspect('equal')

        plt.pause(params.pause)

        if (i < len(t_interp)-1):
            h1.remove()
            h2.remove()
            h3.remove()

    plt.show()

