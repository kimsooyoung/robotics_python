from matplotlib import pyplot as plt
from scipy.optimize import fsolve

import math
import numpy as np
import matrix_helper as mh

def forward_kinematics(l1,l2,theta1,theta2):
    O_01 = [0, 0]
    O_12 = [l1, 0];
    
    # prepping to get homogenous transformations %%
    H_01 = mh.calc_homogeneous_2d(theta1, O_01)
    H_12 = mh.calc_homogeneous_2d(theta2, O_12)
    
    # %%%%%%%% origin  in world frame  %%%%%%
    o = [0, 0];

    # %%%%% end of link1 in world frame %%%%
    P1 = np.array([l1, 0, 1]);
    P1 = np.transpose(P1)
    P0 = H_01 @ P1
    p = [P0[0], P0[1]]
    #
    # %%%% end of link 2 in world frame  %%%%%%%
    Q2 = np.array([l2, 0, 1]);
    Q2 = np.transpose(Q2)
    Q0 = H_01 @ H_12 @ Q2
    q = [Q0[0], Q0[1]]

    return o,p,q

def inverse_kinematics(theta, params):
    
    # forward_kinematics(l1, l2, theta1, theta2)
    o,p,q = forward_kinematics(params[0], params[1], theta[0], theta[1])
    
    # return difference btw ref & end-point
    return q[0] - params[2], q[1] - params[3]
    
def draw_2d_mpl(circle=False):
    # link length
    l1 = 1.0
    l2 = 1.0
    
    if circle is True:
        phi  = np.arange(0,2*np.pi,0.2)
        x_ref_list = 1  + 0.5*np.cos(phi)
        y_ref_list = 0.5+ 0.5*np.sin(phi)
    
    link1, link2 = (None, None)
    cnt = 0
    
    while True:
        
        if circle is False:
            print("=== Type New Ref Points ===")
            x_ref = float(input('x_ref : '))
            y_ref = float(input('y_ref : '))
        else:
            x_ref = x_ref_list[cnt]
            y_ref = y_ref_list[cnt]
            cnt += 1
        
        if link1 is not None:
            link1.remove()
            link2.remove()
            if circle is False:
                point.remove()
        
        parms = [l1, l2, x_ref, y_ref]

        theta = fsolve(inverse_kinematics, [0.01,0.5],parms)
        theta1, theta2 = theta
        print(f"theta1 : {theta1}")
        print(f"theta2 : {theta2}")

        [o,p,q] = forward_kinematics(l1,l2,theta1,theta2)

        # %Draw line from origin to end of link 1
        link1, = plt.plot([o[0], p[0]],[o[1], p[1]],linewidth=5, color='red')

        # %Draw line from end of link 1 to end of link 2
        link2, = plt.plot([p[0], q[0]],[p[1], q[1]],linewidth=5, color='blue')

        # Draw end point
        point, = plt.plot(q[0],q[1],color = 'black',marker = 'o',markersize=5)

        plt.xlabel("x")
        plt.ylabel("y")

        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.grid()
        plt.pause(0.2)
        plt.gca().set_aspect('equal')

        plt.show(block=False)
        # plt.close()
        
if __name__=="__main__":
    try:
        draw_2d_mpl(circle=False)
    except Exception as e:
        print(e)
    finally:
        plt.close()