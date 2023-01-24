import sympy as sy

x = sy.symbols('x', real=True)
f0 = x ** 2 + 2 * x + 1
print(f"f0 : {f0}")

df0_fx = sy.diff(f0, x)
print(f"df0_fx : {df0_fx}")

result = df0_fx.subs(x, 1)
print(f"result : {result}")