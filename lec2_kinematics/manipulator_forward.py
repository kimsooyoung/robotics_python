from matplotlib import pyplot as plt
import matrix_helper as mh
import math
import numpy as np


# define parameters for the two-link
l1 = 1
l2 = 1
theta1 = 0.5;
theta2 = -0.3



# prepping to get homogenous transformations %%
c1 = math.cos(theta1);
c2 = math.cos(theta2);
s1 = math.sin(theta1);
s2 = math.sin(theta2);
O_01 = [0, 0];
O_12 = [l1, 0];

H_01 = mh.calc_homogeneous_2d(theta1, O_01)
H_12 = mh.calc_homogeneous_2d(theta2, O_12)

H_02 = H_01 @ H_12

H01 = np.array([[c1, -s1, O_01[0]],
                [s1, c1,  O_01[1]],
                [0,   0,  1]])
H12 = np.array([[c2, -s2, O_12[0]],
                [s2, c2,  O_12[1]],
                [0,   0,  1]])
H02 = np.matmul(H01,H12)


# %%%%%%%% origin  in world frame  %%%%%%
o = [0, 0];

# %%%%% end of link1 in world frame %%%%
P1 = np.array([l1, 0, 1]);
P1 = np.transpose(P1)
P0 = np.matmul(H_01,P1)
p = [P0[0], P0[1]]
#
# %%%% end of link 2 in world frame  %%%%%%%
Q2 = np.array([l2, 0, 1]);
Q2 = np.transpose(Q2)
Q0 = np.matmul(H_02,Q2)
q = [Q0[0], Q0[1]]

# %Draw line from origin to end of link 1
plt.plot([o[0], p[0]],[o[1], p[1]],linewidth=5, color='red')

# %Draw line from end of link 1 to end of link 2
plt.plot([p[0], q[0]],[p[1], q[1]],linewidth=5, color='blue')

plt.xlabel("x")
plt.ylabel("y")

plt.xlim(-2,2)
plt.ylim(-2,2)
plt.grid()
plt.gca().set_aspect('equal')
# plt.axis('square')

plt.show(block=False)
plt.pause(5)
plt.close()
