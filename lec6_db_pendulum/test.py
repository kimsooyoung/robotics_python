import sympy as sy

x, x_d, x_dd = sy.symbols("x x_d x_dd", real=True)
f = sy.sin(x)
df_dx = sy.diff(f, x) * x_d

f2 = x * x_d
df2_dx = sy.diff(f2, x) * x_d + sy.diff(f2, x_d) * x_dd 
print(sy.simplify(df_dx), sy.simplify(df2_dx))

import numpy as np

a = np.array([
    [2, 0],
    [0, 1]
])
b = np.array([
    [3],
    [2]
])
b2 = np.array([3, 2])
print(a.dot(b))
print(a.dot(b2))