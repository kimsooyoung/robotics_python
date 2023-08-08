import sympy as sy

#define symbolic quantities
theta1,theta2  = sy.symbols('theta1 theta2', real=True)
omega1,omega2  = sy.symbols('omega1 omega2', real=True)
alpha1,alpha2  = sy.symbols('alpha1 alpha2', real=True)
m1,m2,I1,I2,g  = sy.symbols('m1 m2 I1 I2 g', real=True)
c1,c2,l        = sy.symbols('c1 c2 l', real=True)

#1a) position vectors
mpi = sy.pi
cos1 = sy.cos(theta1+mpi/2)
sin1 = sy.sin(theta1+mpi/2)
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

#1b) velocity vectors
q = sy.Matrix([theta1, theta2])
qdot = sy.Matrix([omega1, omega2])
v_G1_x = x_G1.jacobian(q)*qdot
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
# print(L)
# print(type(T))
# print(type(V))
# print(type(L))

#3) Derive equations
qddot = sy.Matrix([alpha1, alpha2])
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

#The equations are of the form M(q)qddot + C(q,qdot)*qdot + G(q) = Bu
M = EOM.jacobian(qddot)
N1 = EOM[0].subs([ (alpha1,0), (alpha2,0)])
N2 = EOM[1].subs([ (alpha1,0), (alpha2,0)])
G1 = N1.subs([ (omega1,0), (omega2,0)])
G2 = N2.subs([ (omega1,0), (omega2,0)])
C1 = N1 - G1
C2 = N2 - G2

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

#linearization
C = sy.Matrix([C1,C2])
G = sy.Matrix([G1,G2])

dGdq =G.jacobian(q)
dGdqdot =G.jacobian(qdot)
dCdq = C.jacobian(q)
dCdqdot = C.jacobian(qdot)

# print(dGdqdot)
# print(dGdq)
# print(dCdq)
# print(dCdqdot)

print('dGdq11 = ', sy.simplify(dGdq[0,0]))
print('dGdq12 = ', sy.simplify(dGdq[0,1]))
print('dGdq21 = ', sy.simplify(dGdq[1,0]))
print('dGdq22 = ', sy.simplify(dGdq[1,1]),'\n')

print('dGdqdot11 = ', sy.simplify(dGdqdot[0,0]))
print('dGdqdot12 = ', sy.simplify(dGdqdot[0,1]))
print('dGdqdot21 = ', sy.simplify(dGdqdot[1,0]))
print('dGdqdot22 = ', sy.simplify(dGdqdot[1,1]),'\n')

print('dCdq11 = ', sy.simplify(dCdq[0,0]))
print('dCdq12 = ', sy.simplify(dCdq[0,1]))
print('dCdq21 = ', sy.simplify(dCdq[1,0]))
print('dCdq22 = ', sy.simplify(dCdq[1,1]),'\n')

print('dCdqdot11 = ', sy.simplify(dCdqdot[0,0]))
print('dCdqdot12 = ', sy.simplify(dCdqdot[0,1]))
print('dCdqdot21 = ', sy.simplify(dCdqdot[1,0]))
print('dCdqdot22 = ', sy.simplify(dCdqdot[1,1]),'\n')
