import sympy as sy

t = sy.symbols('t', real=True)
a0, a1, a2, a3 = sy.symbols('a0 a1 a2 a3', real=True)

pose = a0 + a1*t + a2*t**2 + a3*t**3
vel = a1 + 2*a2*t + 3*a3*t**2
pi = sy.pi

eqn1 = pose.subs(t, 0.0) - (-pi/2 - 0.5)
eqn2 = pose.subs(t, 3.0) - (-pi/2 + 0.5)
eqn3 = vel.subs(t, 0.0) - 0.0
eqn4 = vel.subs(t, 3.0) - 0.0

q = sy.Matrix([a0, a1, a2, a3])
eqn = sy.Matrix([eqn1, eqn2, eqn3, eqn4])

# Ax = b
A = eqn.jacobian(q)
b = -eqn.subs([(a0, 0), (a1, 0), (a2, 0), (a3, 0)])

x = A.inv()*b

print('t1_0 = ', 0.0)
print('t1_N = ', 3.0)
print('\n')
print('a0 = ', x[0])
print('a1 = ', x[1])
print('a2 = ', x[2])
print('a3 = ', x[3])
