import sympy as sy

x,xdot,xddot  = sy.symbols('x xdot xddot', real=True)

# f1 = sin(x), where x = x(t)
f1 = sy.sin(x)
df1dx = sy.diff(f1,x)*xdot
print(df1dx)

#f2 = x*xdot where x = x(t)
f2 = x*xdot
df2dx = sy.diff(f2,x)*xdot + sy.diff(f2,xdot)*xddot
print(df2dx)
