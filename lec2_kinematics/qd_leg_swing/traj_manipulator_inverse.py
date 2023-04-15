import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

from QDTraj import fullTraj
from MatrixHelper import calc_homogeneous_2d

link1, link2 = None, None

class Parameter():
    def __init__(self):
        # define parameters for the two-link
        self.l1 = 300
        self.l2 = 300
        self.O_01 = [0, 0]
        self.O_12 = [self.l1, 0]

def forward_kinematics(l1,l2,theta1,theta2):
    O_01 = [0, 0]
    O_12 = [l1, 0];
    
    # prepping to get homogenous transformations %%
    H_01 = calc_homogeneous_2d(theta1, O_01)
    H_12 = calc_homogeneous_2d(theta2, O_12)
    
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

# def inverse_kinematics(theta, params):
    
#     theta1, theta2 = theta
#     l1, l2, x_ref, y_ref = params
    
#     _, _, q = forward_kinematics(l1, l2, theta1, theta2)
    
#     # return difference btw ref & end-point
#     return q[0] - x_ref, q[1] - y_ref

def inverse_kinematics(theta, l1, l2, x_ref, y_ref):
    
    theta1, theta2 = theta
    _, _, q = forward_kinematics(l1, l2, theta1, theta2)
    
    # return difference btw ref & end-point
    return np.linalg.norm(np.array(q) - np.array([x_ref, y_ref]))
    
def plot(o, p, q):
    
    global link1, link2
    
    if link1 != None:
        link1.remove()
        link2.remove()
    
    plt.xlabel("x")
    plt.ylabel("y")

    plt.xlim(-300,300)
    plt.ylim(-600,100)
    # plt.grid()
    
    plt.gca().set_aspect('equal')
    
    # %Draw line from origin to end of link 1
    link1, = plt.plot([o[0], p[0]],[o[1], p[1]],linewidth=5, color='red')

    # %Draw line from end of link 1 to end of link 2
    link2, = plt.plot([p[0], q[0]],[p[1], q[1]],linewidth=5, color='blue')

    # Draw end point
    point, = plt.plot(q[0],q[1],color = 'black',marker = 'o',markersize=5)

    plt.pause(0.001)
    plt.show(block=False)

def main():
    
    params = Parameter()
    l1, l2 = params.l1, params.l2
    
    points = [
        [-170,  -470],
        [-242, -470],
        [-300, -360],
        [-300, -360],
        [-300, -360],
        [0, -360],
        [0, -360],
        [0, -320],
        [300, -320],
        [300, -320],
        [242, -470],
        [170, -470],
    ]

    num_sample = 30
    traj = fullTraj(points, delta=30, num_sample=num_sample)
    
    for refs in traj:
        
        x_ref, y_ref = refs
        
        thetas = least_squares(
            inverse_kinematics, (np.pi, 0), 
            bounds = ((np.pi, 0), (np.pi * 3/2, np.pi)),
            args=(l1, l2, x_ref, y_ref)
        )
        theta1, theta2 = thetas.x

        o, p, q = forward_kinematics(l1, l2, theta1, theta2)
        
        plot(o ,p, q)
        
if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        plt.close()