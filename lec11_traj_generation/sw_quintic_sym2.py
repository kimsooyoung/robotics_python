import sympy as sy

t = sy.symbols('t', real=True)
t0, tf = sy.symbols('t0 tf', real=True)
q0, qf = sy.symbols('q0 qf', real=True)
a0, a1, a2, a3, a4, a5 = sy.symbols('a0 a1 a2 a3 a4 a5', real=True)

f = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
f_d = sy.diff(f, t)
f_dd = sy.diff(f_d, t)

equ0 = f.subs(t, t0) - q0
equ1 = f.subs(t, tf) - qf
equ2 = f_d.subs(t, t0) - 0
equ3 = f_d.subs(t, tf) - 0
equ4 = f_dd.subs(t, t0) - 0
equ5 = f_dd.subs(t, tf) - 0

q = sy.Matrix([a0, a1, a2, a3, a4, a5])
equ = sy.Matrix([equ0, equ1, equ2, equ3, equ4, equ5])

A = equ.jacobian(q)
b = -equ.subs([
    (a0, 0), (a1, 0), (a2, 0), 
    (a3, 0), (a4, 0), (a5, 0)
])

ans = A.inv() * b

print(f"a0 = {ans[0]}")
print(f"a1 = {ans[1]}")
print(f"a2 = {ans[2]}")
print(f"a3 = {ans[3]}")
print(f"a4 = {ans[4]}")
print(f"a5 = {ans[5]}")