import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

pi = np.pi

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

    H = H_z_theta@H_z_d@H_x_a@H_x_alpha;

    return H

def animate(z):
    
    fig = plt.figure(1)

    for z_temp in z:
        
        ax = p3.Axes3D(fig)
        
        line1, = ax.plot(
            [0, z_temp[0]],
            [0, z_temp[1]],
            [0, z_temp[2]], 
            color='red', linewidth=2
        )
        line2, = ax.plot(
            [z_temp[0], z_temp[3]],
            [z_temp[1], z_temp[4]],
            [z_temp[2], z_temp[5]],
            color='blue', linewidth=2
        )
        line3, = ax.plot(
            [z_temp[3], z_temp[6]],
            [z_temp[4], z_temp[7]],
            [z_temp[5], z_temp[8]],
            color='lightblue', linewidth=2
        )

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        
        # angle 1
        # ax.view_init(elev=90,azim=0)
        
        # angle 2
        ax.view_init(elev=0,azim=-90)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        plt.pause(0.01)
    
    plt.close()

if __name__ == '__main__':
    
    theta1, theta2, theta3 = pi/4, 0, pi/4
    l1, l2, l3 = 0.5, 0.4, 0.25
    
    a1, alpha1, d1 = 0, pi/2, l1
    a2, alpha2, d2 = l2, 0, 0
    a3, alpha3, d3 = l3, 0, 0
    
    N = 50
    
    # # case 1 - theta1
    # theta1_ts = np.linspace(0, pi/4, N)
    # theta2_ts = np.linspace(0, 0, N)
    # theta3_ts = np.linspace(0, 0, N)
    
    # # case 2 - theta2
    # theta1_ts = np.linspace(0, 0, N)
    # theta2_ts = np.linspace(0, pi/4, N)
    # theta3_ts = np.linspace(0, 0, N)
 
    # case 3 - theta3
    theta1_ts = np.linspace(0, 0, N)
    theta2_ts = np.linspace(0, 0, N)
    theta3_ts = np.linspace(0, pi/4, N)
 
 
    z = np.zeros((N, 9))
 
    for i in range(N):
        theta1, theta2, theta3 = theta1_ts[i], theta2_ts[i], theta3_ts[i]

        H01 = DH(a1, alpha1, d1, theta1)
        H12 = DH(a2, alpha2, d2, theta2)
        H23 = DH(a3, alpha3, d3, theta3)

        H01 = H01
        H02 = H01@H12
        H03 = H02@H23
        
        endOfLink1 = H01[0:3,3]
        endOfLink2 = H02[0:3,3]
        endOfLink3 = H03[0:3,3]
        
        z_temp = np.array([*endOfLink1, *endOfLink2, *endOfLink3])
        z[i] = z_temp
    
    animate(z)