import sympy as sy

#define symbolic quantities
theta1, theta2 = sy.symbols("theta1 theta2", real=True)
c1, c2, l = sy.symbols("c1 c2 l", real=True)

H_01 = sy.Matrix([
    [sy.cos(3*sy.pi/2 + theta1), -sy.sin(3*sy.pi/2 + theta1), 0],
    [sy.sin(3*sy.pi/2 + theta1),  sy.cos(3*sy.pi/2 + theta1), 0],
    [0, 0, 1]
])

H_12 = sy.Matrix([
    [sy.cos(theta2), -sy.sin(theta2), l],
    [sy.sin(theta2),  sy.cos(theta2), 0],
    [0, 0, 1]
])

H_02 = H_01 * H_12

G1_1 = sy.Matrix([c1, 0, 1])
G1_0 = H_01 * G1_1

G2_2 = sy.Matrix([c2, 0, 1])
G2_0 = H_02 * G2_2
# End point
E2_2 = sy.Matrix([l, 0, 1])
E2_0 = H_02 * E2_2

G1_0.row_del(2)
G2_0.row_del(2)
E2_0.row_del(2)

q = sy.Matrix([theta1, theta2])
# Jacobian of link1 COM
J_G1 = G1_0.jacobian(q)
# Jacobian of link2 COM
J_G2 = sy.simplify(G2_0.jacobian(q))
# Jacobian of End Point
E_G2 = sy.simplify(E2_0.jacobian(q))
print(J_G1)
print(J_G2)
print(E_G2)