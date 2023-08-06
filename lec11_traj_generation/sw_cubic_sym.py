import sympy as sy

t0, tf = sy.symbols('t0 tf', real=True)
q0, qf = sy.symbols('q0 qf', real=True)

A = sy.Matrix([
        [1, t0, t0**2, t0**3],
        [1, tf, tf**2, tf**3],
        [0,  1, 2*t0,  3*t0**2],
        [0,  1, 2*tf,  3*tf**2]
    ])

b = sy.Matrix([
        [q0],
        [qf],
        [0],
        [0]
    ])

ans = A.inv() * b

print(f'a0 = {ans[0]}')
print(f'a1 = {ans[1]}')
print(f'a2 = {ans[2]}')
print(f'a3 = {ans[3]}')
