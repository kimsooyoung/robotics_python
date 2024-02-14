# ref from : https://www.youtube.com/watch?v=JGk1jsAomDk

import numpy as np
import casadi as ca
import casadi.tools as ca_tools

x = ca.SX.sym('x')
y = ca.SX.sym('y')
z = ca.SX.sym('z')

f = x ** 2 + 100 * z ** 2
g = z + (1 - x) ** 2 - y

lbx = ca.DM([-np.inf, -np.inf, -np.inf])
ubx = ca.DM([np.inf, np.inf, np.inf])
lbg = ca.DM([0])
ubg = ca.DM([0])

prob_struct = { 'f': f, 'x': ca.vertcat(x, y, z), 'g': g }
solver = ca.nlpsol('solver', 'ipopt', prob_struct)

sol = solver(x0=ca.DM([2.5, 3.0, 0.75]), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
print(f"Optimal solution: {sol['x']}")
print(f"Optimal cost: {sol['f']}")