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

parms = parameters()
k1 = parms.k1
k2 = parms.k2
m1 = parms.m1
m2 = parms.m2

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

#compute eigenvalues of uncontrolled system
eigVal,eigVec = np.linalg.eig(A)
print('eig-vals (uncontrolled)=',eigVal) #eigvalues on imaginary axis

#compute controllability of system
Co = control.ctrb(A, B)
print('rank=',np.linalg.matrix_rank(Co))

#pole plaecement
print('\nPole placement');
p = np.array([-1,-2,-3,-4])
K = control.place(A,B,p)
print("K = ",K)
eigVal,eigVec = np.linalg.eig(np.subtract(A,np.matmul(B,K)))
print('eig-vals (controlled)=',eigVal)

#create lqr controller
print('\nLQR');
Q = np.eye((4))
R = 1e-2*np.eye((2));
K, S, E = control.lqr(A,B,Q,R)
print('K=',K)
print('eig-vals (controlled)=',E) #eigvalues are negative
#manually checking eigenvalues
# eigVal,eigVec = np.linalg.eig(np.subtract(A,np.matmul(B,K)))
# print('eig-vals (controlled)=',eigVal)
