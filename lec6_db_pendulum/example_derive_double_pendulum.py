import sympy as sy

#define symbolic quantities
theta1,theta2  = sy.symbols('theta1 theta2', real=True)
omega1,omega2  = sy.symbols('omega1 omega2', real=True)
alpha1,alpha2  = sy.symbols('alpha1 alpha2', real=True)
m1,m2,I1,I2,g  = sy.symbols('m1 m2 I1 I2 g', real=True)
c1,c2,l        = sy.symbols('c1 c2 l', real=True)

#1a) position vectors
mpi = sy.pi
cos1 = sy.cos(3*mpi/2 + theta1)
sin1 = sy.sin(3*mpi/2 + theta1)
H01 = sy.Matrix([ [ cos1, -sin1, 0],
                  [sin1, cos1,  0],
                  [0,     0,     1] ])

cos2 = sy.cos(theta2)
sin2 = sy.sin(theta2)
H12 = sy.Matrix([ [ cos2, -sin2, l],
                  [sin2,  cos2,  0],
                  [0,     0,     1] ])
H02 = H01*H12

C1 = sy.Matrix([c1, 0, 1])
G1 = H01*C1
C2 = sy.Matrix([c2, 0, 1])
G2 = H02*C2

x_G1 = sy.Matrix([G1[0]])
y_G1 = sy.Matrix([G1[1]])
x_G2 = sy.Matrix([G2[0]])
y_G2 = sy.Matrix([G2[1]])
# print(x_G1)
# print(y_G1)
# print(x_G2)
# print(y_G2)

#1b) velocity vectors
q = sy.Matrix([theta1, theta2])
qdot = sy.Matrix([omega1, omega2])
v_G1_x = x_G1.jacobian(q)*qdot #I will teach what jacobian does
v_G1_y = y_G1.jacobian(q)*qdot
v_G2_x = x_G2.jacobian(q)*qdot
v_G2_y = y_G2.jacobian(q)*qdot
v_G1 = sy.Matrix([v_G1_x,v_G1_y])
v_G2 = sy.Matrix([v_G2_x,v_G2_y])

#2) Lagrangian
T = 0.5*m1*v_G1.dot(v_G1) + 0.5*m2*v_G2.dot(v_G2) + \
    0.5*I1*omega1*omega1 + 0.5*I2*(omega1+omega2)*(omega1+omega2)
V = m1*g*G1[1] + m2*g*G2[1]
L = T-V
print(L)
print(f"T: {T}")
print(f"V: {V}")
# print(type(L))

#3) Derive equations
qddot = sy.Matrix([alpha1, alpha2]) #thetaddot
dLdqdot = []
ddt_dLdqdot = []
dLdq = []
EOM = []
mm = len(qddot)
for i in range(0,mm):
    dLdqdot.append(sy.diff(L,qdot[i]))
    tmp = 0;
    for j in range(0,mm):
        tmp += sy.diff(dLdqdot[i],q[j])*qdot[j]+ sy.diff(dLdqdot[i],qdot[j])*qddot[j]
    ddt_dLdqdot.append(tmp)
    dLdq.append(sy.diff(L,q[i]));
    EOM.append(ddt_dLdqdot[i] - dLdq[i])

EOM = sy.Matrix([EOM[0],EOM[1]])
# print(len(EOM))
# print(type(qddot))
# print(type(EOM))

#EOM_0 = A11 theta1ddot + A12 theta2ddot - b1 = 0
#EOM_1 = A21 theta1ddot + A22 theta2ddot - b2 = 0
M = EOM.jacobian(qddot)
b1 = EOM[0].subs([ (alpha1,0), (alpha2,0)])
b2 = EOM[1].subs([ (alpha1,0), (alpha2,0)])
G1 = b1.subs([ (omega1,0), (omega2,0)])
G2 = b2.subs([ (omega1,0), (omega2,0)])
C1 = b1 - G1
C2 = b2 - G2

# C : 코리올리
# G : gravity?
# M(q)*q_dd + C(q, q_d)*q_d + G(q) -Tau = 0
# b = C(q, q_d)*q_d + G(q) -Tau
# G = G(q)
# C = b - G = C(q, q_d)*q_d + G(q) - G(q) = C(q, q_d)*q_d

# A = M
# b = C + G
# print(EOM.shape)
# print(M.shape)
print('M11 = ', sy.simplify(M[0,0]))
print('M12 = ', sy.simplify(M[0,1]))
print('M21 = ', sy.simplify(M[1,0]))
print('M22 = ', sy.simplify(M[1,1]),'\n')

print('C1 = ', sy.simplify(C1))
print('C2 = ', sy.simplify(C2),'\n')
print('G1 = ', sy.simplify(G1))
print('G2 = ', sy.simplify(G2),'\n')
