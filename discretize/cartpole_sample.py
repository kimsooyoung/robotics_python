# ref : https://www.halvorsen.blog/documents/programming/python/resources/powerpoints/Discrete%20Systems%20with%20Python.pdf

import numpy as np
import matplotlib.pyplot as plt
import control

# Parameters defining the system
g = 9.81
M = 0.5
m = 0.2
L = 0.3
I = 0.006
d = 0.1
b = 1 # pendulum up (b=1)
u = 0 # no control
x_init = np.array([0, 0, np.pi - 0.1, 0])

# Simulation Parameters
tstart = 0
tstop = 2
increment = 0.01
t = np.arange(tstart,tstop+1,increment)

# System matrices
# dynamics1
A = np.array([
    [0, 1, 0, 0], 
    [0, -1.0*d/M, 1.0*g*m/M, 0], 
    [0, 0, 0, 1], 
    [0, 1.0*d/(L*M), -1.0*g*(M + m)/(L*M), 0]
])
B = np.array([0, 1/M, 0, -1/(M*L)]).reshape((4,1))

# # dynamics2
# A = np.array([
#     [0, 1, 0, 0], 
#     [0, -1.0*d/M, 1.0*g*m/M, 0], 
#     [0, 0, 0, 1], 
#     [0, -1.0*d/(L*M), 1.0*g*(M + m)/(L*M), 0]
# ])
# B = np.array([0, 1/M, 0, 1/(M*L)]).reshape((4,1))

C = np.array([1, 0, 0, 0]).reshape((1,4))

# A = [[0, 1], [-k/m, -c/m]]
# B = [[0], [1/m]]
# C = [[1, 0]]
sys = control.ss(A, B, C, 0)

# discretization
sysDisc = control.sample_system(sys, increment, method='zoh') 
print(sysDisc)

# Step response for the system
result = control.forced_response(sysDisc, t, u, X0=x_init)

# # Step response for the system
# result = control.forced_response(sys, t, F)
x1 = result.x[0,:]
x2 = result.x[1,:]
x3 = result.x[2,:]
x4 = result.x[3,:]

plt.plot(t, x1, t, x2, t, x3, t, x4)
plt.title('Simulation of Mass-Spring-Damper System')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
plt.show()