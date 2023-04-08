import numpy as np
from matplotlib import pyplot as plt


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

a1 = 1; alpha1 = 0; d1=0; theta1 = np.pi/2;
H01 = DH(a1,alpha1,d1,theta1); #H^0_1

a2 = 1; alpha2 = 0; d2=0; theta2 = np.pi/4;
H12 = DH(a2,alpha2,d2,theta2); #H^1_2

#%Location of joint 1
endOfLink1 = H01[0:3,3];

#Location of joint 2
H02 = np.matmul(H01,H12);
endOfLink2 = H02[0:3,3];


#end-effector position and orientation.
position_of_end_effector = H02[0:3,3];
orientation_of_end_effector = H02[0:3,0:3];
# print(position_of_end_effector)
# print(orientation_of_end_effector)

# %Draw line from end of link 1 to end of link 2
plt.plot([0, endOfLink1[0]],[0, endOfLink1[1]],linewidth=5, color='blue')
plt.plot([endOfLink1[0], endOfLink2[0]],[endOfLink1[1], endOfLink2[1]],linewidth=5, color='red')

# plt.plot([0, endOfLink1[0]],[0, endOfLink1[1]],[0, endOfLink1[2]],linewidth=5, color='blue')
# plt.plot([endOfLink1[0], endOfLink2[0]],[endOfLink1[1], endOfLink2[1]],[endOfLink1[2], endOfLink2[2]],linewidth=5, color='red')

plt.xlabel("x")
plt.ylabel("y")

plt.xlim(-2,2)
plt.ylim(-2,2)
plt.gca().set_aspect('equal')

#
# plt.show(block=False)
# plt.pause(2)
# plt.close()
plt.show()
