import sympy as sy

t0, tf = sy.symbols('t0 tf', real=True)
q0, qf = sy.symbols('q0 qf', real=True)

A = sy.Matrix([ 
        [1, t0],
        [1, tf]
    ])

b = sy.Matrix([
        [q0],
        [qf]
    ])

ans = A.inv() * b
print(f"a0 = {ans[0]}")
print(f"a1 = {ans[1]}")