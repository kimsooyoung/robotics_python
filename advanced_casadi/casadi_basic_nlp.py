# ref from : https://www.youtube.com/watch?v=JGk1jsAomDk

import numpy as np
import casadi as ca
import casadi.tools as ca_tools

x = ca.SX.sym('x')
y = ca.SX.sym('y')
z = y - (1 - x) ** 2

f = x ** 2 + 100 * z ** 2

# f : cost function
# x : decision variable
prob_struct = { 'f': f, 'x': ca.vertcat(x, y) }
solver = ca.nlpsol('solver', 'ipopt', prob_struct)

sol = solver(x0=ca.DM([2.5, 3.0]), lbx=ca.DM([-np.inf, -np.inf]), ubx=ca.DM([np.inf, np.inf]))

print(f"Optimal solution: {sol['x']}")
print(f"Optimal cost: {sol['f']}")