from matplotlib import pyplot as plt
#from scipy.optimize import fsolve
import math
import numpy as np

def cos(theta):
    return np.cos(theta)

def sin(theta):
    return np.sin(theta)

def jacobian_E(l,theta1,theta2):
    J = np.array([[l*(cos(theta1) + cos(theta1 + theta2)), l*cos(theta1 + theta2)], [l*(sin(theta1) + sin(theta1 + theta2)), l*sin(theta1 + theta2)]])
    return J

def forward_kinematics(l,theta1,theta2):
    # prepping to get homogenous transformations %%
    c1 = math.cos(3*np.pi/2+theta1);
    c2 = math.cos(theta2);
    s1 = math.sin(3*np.pi/2+theta1);
    s2 = math.sin(theta2);
    O01 = [0, 0];
    O12 = [l, 0];

    H01 = np.array([[c1, -s1, O01[0]],
                    [s1, c1,  O01[1]],
                    [0,   0,  1]])
    H12 = np.array([[c2, -s2, O12[0]],
                    [s2, c2,  O12[1]],
                    [0,   0,  1]])
    H02 = np.matmul(H01,H12)


    # %%%%%%%% origin  in world frame  %%%%%%
    o = [0, 0];

    # %%%%% end of link1 in world frame %%%%
    P1 = np.array([l, 0, 1]);
    P1 = np.transpose(P1)
    P0 = np.matmul(H01,P1)
    p = [P0[0], P0[1]]
    #
    # %%%% end of link 2 in world frame  %%%%%%%
    Q2 = np.array([l, 0, 1]);
    Q2 = np.transpose(Q2)
    Q0 = np.matmul(H02,Q2)
    q = [Q0[0], Q0[1]]

    #q is the same as e (end effector)
    return o,p,q

l = 1

theta1 = np.pi/2 + np.pi/3;
theta2 = -np.pi/2;
o,p,q = forward_kinematics(l,theta1,theta2)
#print(q) #this is the end-effector position

r = 0.5
center = np.array([q[0]-r,q[1]])


phi  = np.arange(0,2*np.pi,0.2)
x_ref = center[0]  + r*np.cos(phi)
y_ref = center[1]+ r*np.sin(phi)

x_all = []
y_all = []
theta1_all = []
theta2_all = []

for i in range(0,len(phi)):
    #Get the jacobian and its inverse
    J = jacobian_E(l,theta1,theta2)
    #print(J)
    Jinv = np.linalg.inv(J)
    #print(Jinv)

    #Get the errors dx = [x_ref-x, y_ref-y]
    o,p,q = forward_kinematics(l,theta1,theta2)
    x = q[0];
    y = q[1];
    dX = np.array([x_ref[i]-x,y_ref[i]-y])

    #Compute the correction dq = Jinv*dX
    dq = Jinv.dot(dX)

    theta1 += dq[0]
    theta2 += dq[1]

    x_all.append(x)
    y_all.append(y)
    theta1_all.append(theta1)
    theta2_all.append(theta2)

for i in range(len(phi)):
    [o,p,q] = forward_kinematics(l,theta1_all[i],theta2_all[i])

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

    plt.pause(0.05)
    tmp1.remove()
    tmp2.remove()

# plt.show(block=False)
# plt.pause(5)
plt.close()

plt.figure(1)
plt.plot(x_all,y_all,'b--')
plt.plot(x_ref,y_ref,'r-.')
plt.ylabel("y")
plt.xlabel("x")
plt.gca().set_aspect('equal')
plt.show(block=False)
plt.pause(5)
plt.close()
