from matplotlib import pyplot as plt
import numpy as np
import control
import math
from scipy import interpolate
from scipy.integrate import odeint

class parameters:
    def __init__(self):
        self.m1 = 1
        self.m2 = 1
        self.k1 = 2
        self.k2 = 3

        self.pause = 0.01
        self.fps =20

def linear_system(k1, k2, m1, m2):

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

    C = np.array([
                [0,0,1,0],
                [0,0,0,1]
                ])

    return A, B, C

if __name__=='__main__':
    
    parms = parameters()
    
    k1, k2, m1, m2 = parms.k1, parms.k2, parms.m1, parms.m2
    A, B, C = linear_system(k1, k2, m1, m2)

    # compute eigenvalues of uncontrolled system
    eigVal, eigVec = np.linalg.eig(A)
    print('eig-vals (uncontrolled)=') #eigvalues on imaginary axis
    print(eigVal, '\n')

    # compute observability of the system (2 ways)
    Ob = control.obsv(A,C)
    print('rank=',np.linalg.matrix_rank(Ob))
    Ob_alt = control.ctrb(A.T, C.T) #alternate way of computing observability

    # To check if obsv(A,C) gives same values as ctrl(A',C')
    print(Ob)
    print(Ob_alt.T)

    #pole placement
    print('\nPole placement');
    p = np.array([-5,-5.5,-6.5,-6])
    # caution! Matirix form requires certain form
    # L_trans = control.place(A, C, p)
    L_trans = control.place(A.T, C.T, p)
    L = L_trans.T
    print("L = ",L)
    
    # eigVal, eigVec = np.linalg.eig(np.subtract(A,np.matmul(L,C)))
    eigVal, eigVec = np.linalg.eig(A - L@C)
    print('eig-vals (controlled)=', eigVal)
    
    # Review: controllability
    Co = control.ctrb(A,B)
    print('rank=',np.linalg.matrix_rank(Co))
    
    K = control.place(A,B,p)
    eigVal, eigVec = np.linalg.eig(A - B@K)
    print('eig-vals (controlled)=', eigVal)
    