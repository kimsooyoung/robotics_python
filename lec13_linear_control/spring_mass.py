from matplotlib import pyplot as plt
import numpy as np
import control
import math
from scipy import interpolate
from scipy.integrate import odeint

class parameters:
    def __init__(self):
        # two masses
        self.m1 = 1
        self.m2 = 1

        # spring constants
        self.k1 = 2
        self.k2 = 3

        # simulation parameters
        self.pause = 0.01
        self.fps =20

if __name__=='__main__':
    parms = parameters()

    k1, k2 = parms.k1, parms.k2
    m1, m2 = parms.m1, parms.m2

    A = np.array([
        [0,0,1,0],
        [0,0,0,1],
        [-(k1/m1+k2/m1), k2/m1, 0, 0],
        [k2/m2, -k2/m2, 0, 0]
    ])

    B = np.array([
        [0,0],
        [0,0],
        [-1/m1, 0],
        [1/m2,1/m2]
    ])

    # 1. calculate eigenvalues of uncontrolled system
    #    eigenvalues should have negative-real part
    # compute eigenvalues of uncontrolled system
    eigVal,eigVec = np.linalg.eig(A)
    print('eig-vals (uncontrolled)=',eigVal) #eigvalues on imaginary axis

    # 2. check controllability of system
    #   rank of controllability matrix 
    #   should be equal to number of states
    #compute controllability of system
    Co = control.ctrb(A, B)
    print('rank=',np.linalg.matrix_rank(Co))

    # 3. pole plaecement
    #    place poles at desired locations
    #    u = -k * x
    #    x_d = A*x + B*u = A*x - B*k*x = (A-B*k)*x
    print('\nPole placement');
    p = np.array([-1,-2,-3,-4])
    K = control.place(A,B,p)
    print("K = ",K)
    # simpler one 
    eigVal,eigVec = np.linalg.eig(A - B@K)
    print('eig-vals (controlled)=',eigVal)

    #create lqr controller
    print('\nLQR')
    Q = np.eye((4))
    R = 1e-2 * np.eye((2))
    # E = eig(A - B@K)
    K, S, E = control.lqr(A,B,Q,R)
    print('K=',K) # K : gain matrix
    print('S=',S) # S : solution to Riccati equation
    print('eig-vals  (controlled)=',E) # E : eigenvalues of closed loop system
