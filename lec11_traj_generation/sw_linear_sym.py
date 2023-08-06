import sympy as sy

# q0 = a0 + a1*t0
# qf = a0 + a1*tf
#
# Ax = b

a0, a1 = sy.symbols('a0 a1')
q0, qf = sy.symbols('q0 qf')
t0, tf = sy.symbols('t0 tf')

A = sy.Matrix([
    [1, t0],
    [1, tf]
])

b = sy.Matrix([
    [q0],
    [qf]
])

x = A.inv() * b
print(f"a0 = {x[0]}")
print(f"a1 = {x[1]}")
