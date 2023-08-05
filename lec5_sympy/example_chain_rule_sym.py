import sympy as sy

x, x_d, x_dd = sy.symbols('x x_d x_dd', real=True)

# ex1
f1 = sy.sin(x)
df1dx = sy.diff(f1, x) * x_d
print(df1dx)

# ex2
f2 = x * x_d
df2fx = sy.diff(f2, x) * x_d + sy.diff(f2, x_d) * x_dd
print(df2fx)
