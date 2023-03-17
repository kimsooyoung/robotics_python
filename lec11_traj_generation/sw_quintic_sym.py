import sympy as sy

t0, tf = sy.symbols('t0 tf', real=True)
q0, qf = sy.symbols('q0 qf', real=True)

A = sy.Matrix([ 
        [1, t0, t0**2, t0**3, t0**4, t0**5],
        [1, tf, tf**2, tf**3, tf**4, tf**5],
        [0,  1, 2*t0,  3*t0**2, 4*t0**3, 5*t0**4],
        [0,  1, 2*tf,  3*tf**2, 4*tf**3, 5*tf**4],
        [0,  0, 2,  6*t0, 12*t0**2, 20*t0**3],
        [0,  0, 2,  6*tf, 12*tf**2, 20*tf**3]
    ])

b = sy.Matrix([
        [q0],
        [qf],
        [0],
        [0],
        [0],
        [0]
    ])

ans = A.inv() * b

print(f"a0 = {ans[0]}")
print(f"a1 = {ans[1]}")
print(f"a2 = {ans[2]}")
print(f"a3 = {ans[3]}")
print(f"a4 = {ans[4]}")
print(f"a5 = {ans[5]}")