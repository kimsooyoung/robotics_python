# ref : https://www.halvorsen.blog/documents/programming/python/resources/powerpoints/Discrete%20Systems%20with%20Python.pdf

import numpy as np
import matplotlib.pyplot as plt
import control

# Parameters defining the system
c = 4 # Damping constant
k = 2 # Stiffness of the spring
m = 20 # Mass
F = 5 # Force

# Simulation Parameters
tstart = 0
tstop = 60
increment = 0.1
t = np.arange(tstart,tstop+1,increment)

# System matrices
A = [[0, 1], [-k/m, -c/m]]
B = [[0], [1/m]]
C = [[1, 0]]
sys = control.ss(A, B, C, 0)

# discretization
sysDisc = control.sample_system(sys, increment, method='zoh') 
print(sysDisc)

# Step response for the system
result = control.forced_response(sysDisc, t, F)

# # Step response for the system
# result = control.forced_response(sys, t, F)
x1 = result.x[0,:]
x2 = result.x[1,:]

plt.plot(t, x1, t, x2)
plt.title('Simulation of Mass-Spring-Damper System')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
plt.show()