# ref from : https://www.youtube.com/watch?v=JGk1jsAomDk

import numpy as np
import casadi as ca
import casadi.tools as ca_tools

c = ca.DM([-7000, -6000])
A = ca.DM([
    [4000, 3000], 
    [60, 80]
])
b = ca.DM([100000, 2000])

lb = ca.DM([0, 0])
ub = ca.DM([np.inf, np.inf])
x = ca.SX.sym('x', 2)
e = ca.mtimes(A, x) - b
f1 = ca.mtimes(c.T, x)
prob_struct = {'f': f1, 'x': x, 'g': e}
solver = ca.nlpsol('solver', 'ipopt', prob_struct)
# or
# solver = ca.qpsol('solver', 'qpoases', prob_struct)

# takes the parameter value (p), the bounds (lbx, ubx, lbg, ubg) 
# and a guess for the primal-dual solution (x0, lam_x0, lam_g0) 
# and returns the optimal solution.
sol = solver(x0=ca.DM([1, 1]), lbx=lb, ubx=ub, lbg=ca.DM([-np.inf, -np.inf]), ubg=ca.DM([0, 0]))
print(f"Optimal solution: {sol['x']}")
print(f"Optimal cost: {sol['f']}")

