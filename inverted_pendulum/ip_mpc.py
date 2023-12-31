import numpy as np
import matplotlib.pyplot as plt
from control import place
from scipy import integrate

m = 1
M = 5
L = 2
g = -10
d = 1

b = 1 # pendulum up (b=1)

A = np.array([[0,1,0,0],\
              [0,-d/M,b*m*g/M,0],\
              [0,0,0,1],\
              [0,-b*d/(M*L),-b*(m+M)*g/(M*L),0]])

B = np.array([0,1/M,0,b/(M*L)]).reshape((4,1))

print(np.linalg.eig(A)[0])       # Eigenvalues
print(np.linalg.det(ctrb(A,B)))  # Determinant of controllability matrix
