import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

def cos(angle):
    return np.cos(angle)

def sin(angle):
    return np.sin(angle)

def DH(a,alpha,d,theta):

    cth = cos(theta);
    sth = sin(theta);
    cal = cos(alpha);
    sal = sin(alpha);

    H_z_theta = np.array([
                   [cth, -sth, 0, 0],
                   [sth,  cth, 0, 0],
                   [0 ,    0, 1, 0],
                   [0 ,    0, 0, 1]]);

    H_z_d = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, d],
                [0, 0, 0, 1]]);

    H_x_a = np.array([
                    [1, 0, 0, a],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]);

    H_x_alpha = np.array([
                  [1,   0,   0,   0],
                  [0,  cal, -sal, 0],
                  [0,  sal,  cal, 0],
                  [0,   0,    0,  1]]);

    H_z = np.matmul(H_z_theta,H_z_d);
    H_x = np.matmul(H_x_a,H_x_alpha);
    H = np.matmul(H_z,H_x);
    # H = H_z_theta*H_z_d*H_x_a*H_x_alpha;

    return H

if __name__ == '__main__':
    
    # a1, alpha1, d1, theta1 = 0, 0, 0.5, np.pi/3
    a1, alpha1, d1, theta1 = 0, 0, 0.5, 0
    H01 = DH(a1,alpha1,d1,theta1); #H^0_1

    # a2, alpha2, d2, theta2 = 0, -np.pi/2, 0.4, 0
    a2, alpha2, d2, theta2 = 0, 0, 0.4, 0
    H12 = DH(a2,alpha2,d2,theta2); #H^1_2

    a3, alpha3, d3, theta3 = 0, 0, 0.25, 0
    H23 = DH(a3,alpha3,d3,theta3); #H^2_3

    #%Location of joint 1
    endOfLink1 = H01[0:3,3];
    # print(endOfLink1)

    #Location of joint 2
    H02 = H01@H12
    endOfLink2 = H02[0:3,3];
    # print(endOfLink2)

    #Location of joint 3
    H03 = H02@H23
    endOfLink3 = H03[0:3,3];
    # print(endOfLink3)

    #end-effector position and orientation.
    position_of_end_effector = H03[0:3,3];
    orientation_of_end_effector = H03[0:3,0:3];
    print(position_of_end_effector)
    print(orientation_of_end_effector)

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    line1, = ax.plot(
        [0, endOfLink1[0]],
        [0, endOfLink1[1]],
        [0, endOfLink1[2]], 
        color='red', linewidth=2
    )
    line2, = ax.plot(
        [endOfLink1[0], endOfLink2[0]],
        [endOfLink1[1], endOfLink2[1]],
        [endOfLink1[2], endOfLink2[2]],
        color='blue', linewidth=2
    )
    line3, = ax.plot(
        [endOfLink2[0], endOfLink3[0]],
        [endOfLink2[1], endOfLink3[1]],
        [endOfLink2[2], endOfLink3[2]],
        color='lightblue', linewidth=2
    )

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(azim=64,elev=29)
    
    plt.show()
    