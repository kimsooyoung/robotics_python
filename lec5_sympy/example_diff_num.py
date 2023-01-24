import sympy as sy

x  = sy.symbols('x', real=True) # defining the variables
f0 = x**2+2*x+1
print(f0)

xval = 1
pert = 1e-5
F0 = f0.subs(x,xval)
F1 = f0.subs(x,xval-pert)
F2 = f0.subs(x,xval+pert)

#forward difference
dfdx_num = (F2-F0)/pert
print(dfdx_num)

#central difference
dfdx_num = (F2-F1)/(2*pert)
print(dfdx_num)
