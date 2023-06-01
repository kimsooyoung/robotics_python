import sympy as sy


l = sy.symbols('l')

A = sy.Matrix([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 1/(2*l), 0, -1/(2*l)],
])

print(A.inv())