import sympy as sy
import numpy as np
import math

def cos(theta):
    return np.cos(theta)

def sin(theta):
    return np.sin(theta)

def jacobian_E(l, theta1, theta2):
    J = np.array([
        [l*(cos(theta1) + cos(theta1 + theta2)), l*cos(theta1 + theta2)], 
        [l*(sin(theta1) + sin(theta1 + theta2)), l*sin(theta1 + theta2)]
    ])
    return J

def forward_kinematics(l,theta1,theta2):
    # prepping to get hforward_kinematicsomogenous transformations %%
    c1 = np.cos(3 * np.pi/2 + theta1)
    c2 = np.cos(theta2)
    s1 = np.sin(3 * np.pi/2 + theta1)
    s2 = np.sin(theta2)
    
    O01 = [0, 0]
    O12 = [l, 0]

    H01 = np.array([[c1, -s1, O01[0]],
                    [s1, c1,  O01[1]],
                    [0,   0,  1]])
    H12 = np.array([[c2, -s2, O12[0]],
                    [s2, c2,  O12[1]],
                    [0,   0,  1]])
    H02 = np.matmul(H01, H12)
    
    # %%%%%%%% origin  in world frame  %%%%%%
    o = [0, 0];

    # %%%%% end of link1 in world frame %%%%
    P1 = np.array([l, 0, 1])
    P1 = np.transpose(P1)
    P0 = np.matmul(H01, P1)
    p = [P0[0], P0[1]]
    #
    # %%%% end of link 2 in world frame  %%%%%%%
    Q2 = np.array([l, 0, 1]);
    Q2 = np.transpose(Q2)
    Q0 = np.matmul(H02, Q2)
    q = [Q0[0], Q0[1]]

    #q is the same as e (end effector)
    return o, p, q