import sympy as sy

x, y, z = sy.symbols('x y z')
xd, yd, zd = sy.symbols('xd yd zd')

q = sy.Matrix([x, y, z])
qd = sy.Matrix([xd, yd, zd])

A = sy.Matrix([
    x**2 + y**2 + z**2,
    x*y + y*z + z*x,
])

J = A.jacobian(q)
col, row = J.shape

print(J[0,:])
print(J[0,0])

J_dot = []

for i in range(col):
    J_temp = J[i,:].jacobian(q) * qd
    J_dot.append(list(J_temp))
    
print(J_dot)

print(J_dot[0][0])