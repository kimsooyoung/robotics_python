from matplotlib import pyplot as plt
import matrix_helper as mh
import math
import numpy as np


# define parameters for the two-link
l1 = 1
l2 = 1
O_01 = [0, 0];
O_12 = [l1, 0];

def draw_2d_mpl():
    link1 = None
    link2 = None

    while True:
        theta1 = float(input('theta1 : '))
        theta2 = float(input('theta2 : '))

        if link1 is not None:
            link1.remove()
            link2.remove()

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

        # %Draw line from origin to end of link 1
        link1, = plt.plot([o[0], p[0]],[o[1], p[1]],linewidth=5, color='red')

        # %Draw line from end of link 1 to end of link 2
        link2, = plt.plot([p[0], q[0]],[p[1], q[1]],linewidth=5, color='blue')

        plt.xlabel("x")
        plt.ylabel("y")

        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.grid()
        plt.gca().set_aspect('equal')
        # plt.axis('square')
        plt.pause(0.3)

        plt.show(block=False)

        print("Type New Thetas")

if __name__=="__main__":
    try:
        draw_2d_mpl()
    except Exception as e:
        print(e)
    finally:
        plt.close()
