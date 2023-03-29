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

C = np.array([
             [0,0,1,0],
             [0,0,0,1]
             ])

#compute eigenvalues of uncontrolled system
eigVal,eigVec = np.linalg.eig(A)
print('eig-vals (uncontrolled)=') #eigvalues on imaginary axis
for i in np.arange(0,4):
    print(eigVal[i])

#compute observability of the system (2 ways)
Ob = control.obsv(A,C)
print('rank=',np.linalg.matrix_rank(Ob))
Ob_alt = control.ctrb(np.transpose(A),np.transpose(C)) #alternate way of computing observability

#To check if obsv(A,C) gives same values as ctrl(A',C')
# print(Ob)
# print(np.transpose(Ob_alt))

#pole placement
print('\nPole placement');
p = np.array([-5,-5.5,-6.5,-6])
L_trans = control.place(np.transpose(A),np.transpose(C),p)
L = np.transpose(L_trans)
print("L = ",L)
eigVal,eigVec = np.linalg.eig(np.subtract(A,np.matmul(L,C)))
print('eig-vals (controlled)=')
for i in np.arange(0,4):
    print(eigVal[i])
