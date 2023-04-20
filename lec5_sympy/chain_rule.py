import sympy as sy

x, x_d, x_dd = sy.symbols('x x_d x_dd', real=True)
f0 = sy.sin(x**2)

df0_fx = sy.diff(f0, x)
print(f"df0_fx : {df0_fx}")

f1 = x * x_d
df1_fx = sy.diff(f1, x) * x_d + sy.diff(f1, x_d) * x_dd
print(f"df1_fx : {df1_fx}")