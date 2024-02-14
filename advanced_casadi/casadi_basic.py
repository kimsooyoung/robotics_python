# ref from : https://www.youtube.com/watch?v=JGk1jsAomDk

import numpy as np
import casadi as ca
import casadi.tools as ca_tools

# (size 2x2 matrix A)  @ (size 2 vector x) 
A = ca.SX.sym('A', 2, 2)
x = ca.SX.sym('x', 2)
b = ca.mtimes(A, ca.sin(x))
print(b)

# Function example - sphere volume
r = ca.SX.sym('r')
V = 4/3 * ca.pi * r**3

# 4 * pi * r^2
A = ca.jacobian(V, r)
f = ca.Function('f', [r], [A])
print(f(2))