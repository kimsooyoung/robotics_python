from matplotlib import pyplot as plt
import pendulum_helper as ph
#from scipy.optimize import fsolve

import math
import numpy as np

class Parameter():
    def __init__(self):
        self.l = 1.0
        self.theta1 = np.pi/2 + np.pi/3
        self.theta2 = -np.pi/2
        
        self.r = 0.5
        self.step_size = 0.2

def generate_path(params, q):
    
    phi = np.arange(0, 2*np.pi, params.step_size)
    
    # first reference point == robot initial point
    # center = np.array([q[0] - params.r, q[1]])
    center = np.array([q[0], q[1]])
    x_ref = center[0] + params.r * np.cos(phi)
    y_ref = center[1] + params.r * np.sin(phi)

    return x_ref, y_ref

def simualtion(params, x_ref, y_ref):
    
    # get initial states     
    theta1, theta2 = params.theta1, params.theta2

    x_all = []
    y_all = []
    theta1_all = []
    theta2_all = []
    
    phi  = np.arange(0, 2*np.pi, params.step_size)

    for i in range(0,len(phi)):
        # 1. Get the jacobian and its inverse
        J = ph.jacobian_E(params.l, theta1, theta2)
        # 2. Calculate J inv
        Jinv = np.linalg.inv(J)
        # 3. Get current endpoint then calculate the errors dX
        _, _, q = ph.forward_kinematics(params.l, theta1, theta2)
        x = q[0]
        y = q[1]
        
        dX = np.array([x_ref[i] - x , y_ref[i] - y])

        # 4. Compute the correction dq
        dq = Jinv.dot(dX)

        # 5. update joint positions
        theta1 += dq[0]
        theta2 += dq[1]

        # 6. append data for output results
        x_all.append(x)
        y_all.append(y)
        theta1_all.append(theta1)
        theta2_all.append(theta2)
        
    return x_all, y_all, theta1_all, theta2_all


def animation(params, x_all, y_all, theta1_all, theta2_all):
    phi  = np.arange(0, 2*np.pi, params.step_size)

    for i in range(len(phi)):
        [o, p, q] = ph.forward_kinematics(params.l, theta1_all[i], theta2_all[i])

        plt.plot(q[0],q[1],color = 'black',marker = 'o',markersize=5)

        # %Draw line from origin to end of link 1
        tmp1, = plt.plot([o[0], p[0]],[o[1], p[1]],linewidth=5, color='red')

        # %Draw line from end of link 1 to end of link 2
        tmp2, = plt.plot([p[0], q[0]],[p[1], q[1]],linewidth=5, color='blue')

        plt.xlabel("x")
        plt.ylabel("y")

        # plt.grid()
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.gca().set_aspect('equal')

        plt.pause(0.01)
        tmp1.remove()
        tmp2.remove()

    plt.close()
    plt.figure(1)
    
    plt.plot(x_all, y_all, 'b--')
    plt.plot(x_ref, y_ref, 'r-.')
    plt.ylabel("y")
    plt.xlabel("x")
    
    plt.gca().set_aspect('equal')
    
    plt.show(block=False)
    plt.pause(5)
    plt.close()

if __name__=="__main__":
    
    params = Parameter()
    # get initial points
    o, p, q = ph.forward_kinematics(params.l, params.theta1, params.theta2)
    
    # generate path
    x_ref, y_ref = generate_path(params, q)
    
    # 1. calculate forward Kinematics with state vectors 
    # 2. Get update vector using Jacobian
    # 3. Update joint position then FK again
    x_all, y_all, theta1_all, theta2_all = simualtion(params, x_ref, y_ref)
    
    # motion animation then plotting traj
    animation(params, x_all, y_all, theta1_all, theta2_all)