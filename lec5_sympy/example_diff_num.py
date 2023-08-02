import sympy as sy

# prepare equations
x = sy.symbols('x', real=True)
f0 = x ** 2 + 2 * x + 1
print(f'f0 : {f0}')

x_val = 1
epsilon = 1e-5
F0 = f0.subs(x, x_val - epsilon)
F1 = f0.subs(x, x_val)
F2 = f0.subs(x, x_val + epsilon)

# calculate numerical diff
fwd_diff = (F2 - F1) / epsilon
ctr_diff = (F2 - F0) / (2 * epsilon)
print(f'fwd_diff : {fwd_diff} / ctr_diff : {ctr_diff}')
