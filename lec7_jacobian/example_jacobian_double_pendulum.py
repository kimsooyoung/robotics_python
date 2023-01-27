import sympy as sy

#define symbolic quantities
theta1,theta2  = sy.symbols('theta1 theta2', real=True)
c1,c2,l        = sy.symbols('c1 c2 l', real=True)

mpi = sy.pi
cos1 = sy.cos(3*mpi/2 + theta1)
sin1 = sy.sin(3*mpi/2 + theta1)
H01 = sy.Matrix([ [ cos1, -sin1, 0],
         [sin1, cos1,  0],
         [0,     0,     1] ])

cos2 = sy.cos(theta2)
sin2 = sy.sin(theta2)
H12 = sy.Matrix([ [ cos2, -sin2, l],
         [sin2, cos2,  0],
         [0,     0,     1] ])
H02 = H01*H12

#first index is associated with the name and second index with frame (e.g, G2 in frame 0 is G2_0)
G1_1 = sy.Matrix([c1, 0, 1])
G1_0 = H01*G1_1
G2_2 = sy.Matrix([c2, 0, 1])
G2_0 = H02*G2_2
E_2 = sy.Matrix([l, 0, 1])
E_0 = H02*E_2

G1_0.row_del(2)
G2_0.row_del(2)
E_0.row_del(2)

theta = sy.Matrix([theta1, theta2])
J_G1 = G1_0.jacobian(theta)
J_G2 = sy.simplify(G2_0.jacobian(theta))
J_E = sy.simplify(E_0.jacobian(theta))
print(J_G1)
print(J_G2)
print(J_E)
