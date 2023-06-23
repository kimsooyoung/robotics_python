import sympy as sy

q1, q2, q3, q4 = sy.symbols('q1 q2 q3 q4')
u1, u2, u3, u4 = sy.symbols('u1 u2 u3 u4')
a1, a2, a3, a4 = sy.symbols('a1 a2 a3 a4')
m1, m2, m3, m4 = sy.symbols('m1 m2 m3 m4')
I1, I2, I3, I4 = sy.symbols('I1 I2 I3 I4')
l1, l2, l3, l4 = sy.symbols('l1 l2 l3 l4')
lx, ly = sy.symbols('lx ly')
g = sy.symbols('g')

q = sy.Matrix([q1, q2, q3, q4])
qdot = sy.Matrix([u1, u2, u3, u4])
qddot = sy.Matrix([a1, a2, a3, a4])

### position vectors ###
def Homogeneous(angle, x, y):
    return sy.Matrix([
        [sy.cos(angle), -sy.sin(angle), x],
        [sy.sin(angle), sy.cos(angle),  y],
        [0,   0,  1]
    ])
    
H01 = Homogeneous( 3*sy.pi/2 + q1, 0, 0)
H12 = Homogeneous( q2, l1, 0)
H02 = H01*H12

H03 = Homogeneous( 3*sy.pi/2 + q3, lx, ly)
H34 = Homogeneous( q4, l3, 0)
H04 = H03*H34

G1 = H01 * sy.Matrix([0.5*l1, 0, 1])
G2 = H02 * sy.Matrix([0.5*l2, 0, 1])
G3 = H03 * sy.Matrix([0.5*l3, 0, 1])
G4 = H04 * sy.Matrix([0.5*l4, 0, 1])

G1_xy = sy.Matrix([G1[0], G1[1]])
G2_xy = sy.Matrix([G2[0], G2[1]])
G3_xy = sy.Matrix([G3[0], G3[1]])
G4_xy = sy.Matrix([G4[0], G4[1]])

### velocity vectors ###
v_G1 = G1_xy.jacobian(q) * qdot
v_G2 = G2_xy.jacobian(q) * qdot
v_G3 = G3_xy.jacobian(q) * qdot
v_G4 = G4_xy.jacobian(q) * qdot

### end of last link is P2 and P4 (say) then position and velocity is given by ###
P2 = H02 * sy.Matrix([l2, 0, 1])
P4 = H04 * sy.Matrix([l4, 0, 1])
del_x = sy.simplify(P2[0] - P4[0])
del_y = sy.simplify(P2[1] - P4[1])

del_xy = sy.Matrix([del_x, del_y])
del_v = del_xy.jacobian(q) * qdot

om1 = u1
om2 = om1 + u2
om3 = u3
om4 = om3 + u4

### lagrangian ###
T = 0.5*m1*v_G1.dot(v_G1) + \
    0.5*m2*v_G2.dot(v_G2) + \
    0.5*m3*v_G3.dot(v_G3) + \
    0.5*m4*v_G4.dot(v_G4) + \
    0.5*I1*om1**2 + \
    0.5*I2*om2**2 + \
    0.5*I3*om3**2 + \
    0.5*I4*om4**2

V = m1*g*G1[1] + m2*g*G2[1] + m3*g*G3[1] + m4*g*G4[1]

L = T - V

dL_dq_d = []
dt_dL_dq_d = []
dL_dq = []
EOM = []

for i in range(len(q)):
    dL_dq_d.append(sy.diff(L, qdot[i]))
    temp = 0
    for j in range(len(q)):
        temp += sy.diff(dL_dq_d[i], q[j]) * qdot[j] + \
                sy.diff(dL_dq_d[i], qdot[j]) * qddot[j]
    
    dt_dL_dq_d.append(temp)
    dL_dq.append(sy.diff(L, q[i]))
    
    EOM.append(dt_dL_dq_d[i] - dL_dq[i])

EOM = sy.Matrix([EOM[0],EOM[1],EOM[2],EOM[3]])

### collecting equations as A a = b ###

M = EOM.jacobian(qddot)
b = EOM.subs([
    (a1, 0),
    (a2, 0),
    (a3, 0),
    (a4, 0)
])
G = b.subs([
    (u1, 0),
    (u2, 0),
    (u3, 0),
    (u4, 0)
])
C = b - G

J = del_xy.jacobian(q)
col, row = J.shape
Jdot = []

for i in range(col):
    J_temp = J[i,:].jacobian(q) * qdot
    Jdot.append(list(J_temp))
    

with open("fourlinkchain_rhs.py", "w") as f:
    
    f.write("import numpy as np \n\n")
    f.write("def cos(angle): \n")
    f.write("    return np.cos(angle) \n\n")
    f.write("def sin(angle): \n")
    f.write("    return np.sin(angle) \n\n")
    
    f.write("def fourlinkchain(z, t, params): \n\n")
    f.write("    q1, u1 = z[0], z[1] \n")
    f.write("    q2, u2 = z[2], z[3] \n")
    f.write("    q3, u3 = z[4], z[5] \n")
    f.write("    q4, u4 = z[6], z[7] \n\n")
    
    f.write("    g, lx, ly = params.g, params.lx, params.ly \n")
    f.write("    m1, m2, m3, m4 = params.m1, params.m2, params.m3, params.m4 \n")
    f.write("    I1, I2, I3, I4 = params.I1, params.I2, params.I3, params.I4 \n")
    f.write("    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4 \n\n")
    
    f.write(f"    M11 = {sy.simplify(M[0,0])} \n\n")
    f.write(f"    M12 = {sy.simplify(M[0,1])} \n\n")
    f.write(f"    M13 = {sy.simplify(M[0,2])} \n\n")
    f.write(f"    M14 = {sy.simplify(M[0,3])} \n\n")
    
    f.write(f"    M21 = {sy.simplify(M[1,0])} \n\n")
    f.write(f"    M22 = {sy.simplify(M[1,1])} \n\n")
    f.write(f"    M23 = {sy.simplify(M[1,2])} \n\n")
    f.write(f"    M24 = {sy.simplify(M[1,3])} \n\n")
    
    f.write(f"    M31 = {sy.simplify(M[2,0])} \n\n")
    f.write(f"    M32 = {sy.simplify(M[2,1])} \n\n")
    f.write(f"    M33 = {sy.simplify(M[2,2])} \n\n")
    f.write(f"    M34 = {sy.simplify(M[2,3])} \n\n")
    
    f.write(f"    M41 = {sy.simplify(M[3,0])} \n\n")
    f.write(f"    M42 = {sy.simplify(M[3,1])} \n\n")
    f.write(f"    M43 = {sy.simplify(M[3,2])} \n\n")
    f.write(f"    M44 = {sy.simplify(M[3,3])} \n\n")
    
    f.write(f"    C1 = {sy.simplify(C[0])}\n\n")
    f.write(f"    C2 = {sy.simplify(C[1])}\n\n")
    f.write(f"    C3 = {sy.simplify(C[2])}\n\n")
    f.write(f"    C4 = {sy.simplify(C[3])}\n\n")
    
    f.write(f"    G1 = {sy.simplify(G[0])}\n\n")
    f.write(f"    G2 = {sy.simplify(G[1])}\n\n")
    f.write(f"    G3 = {sy.simplify(G[2])}\n\n")
    f.write(f"    G4 = {sy.simplify(G[3])}\n\n")
    
    f.write(f"    J11 = {J[0,0]}\n\n")
    f.write(f"    J12 = {J[0,1]}\n\n")
    f.write(f"    J13 = {J[0,2]}\n\n")
    f.write(f"    J14 = {J[0,3]}\n\n")
    
    f.write(f"    J21 = {J[1,0]}\n\n")
    f.write(f"    J22 = {J[1,1]}\n\n")
    f.write(f"    J23 = {J[1,2]}\n\n")
    f.write(f"    J24 = {J[1,3]}\n\n")
    
    f.write(f"    Jdot11 = {Jdot[0][0]}\n\n")
    f.write(f"    Jdot12 = {Jdot[0][1]}\n\n")
    f.write(f"    Jdot13 = {Jdot[0][2]}\n\n")
    f.write(f"    Jdot14 = {Jdot[0][3]}\n\n")
    
    f.write(f"    Jdot21 = {Jdot[1][0]}\n\n")
    f.write(f"    Jdot22 = {Jdot[1][1]}\n\n")
    f.write(f"    Jdot23 = {Jdot[1][2]}\n\n")
    f.write(f"    Jdot24 = {Jdot[1][3]}\n\n")
    
    f.write("    A = np.array([ \n")
    f.write(f"        [M11, M12, M13, M14], \n")
    f.write(f"        [M21, M22, M23, M24], \n")
    f.write(f"        [M31, M32, M33, M34], \n")
    f.write(f"        [M41, M42, M43, M44] \n")
    f.write("    ]) \n\n")
    
    f.write("    b = -np.array([ \n")
    f.write(f"        [C1 + G1], \n")
    f.write(f"        [C2 + G2], \n")
    f.write(f"        [C3 + G3], \n")
    f.write(f"        [C4 + G4] \n")
    f.write("    ]) \n\n")
    
    f.write(f"    J = np.array([\n")
    f.write(f"        [J11, J12, J13, J14],\n")
    f.write(f"        [J21, J22, J23, J24],\n")
    f.write(f"    ])\n\n")

    f.write(f"    Jdot = np.array([\n")
    f.write(f"        [Jdot11, Jdot12, Jdot13, Jdot14],\n")
    f.write(f"        [Jdot21, Jdot22, Jdot23, Jdot24],\n")
    f.write(f"    ])\n\n")
    
    f.write(f"    qdot = np.array([u1, u2, u3, u4])\n\n")
    
    f.write(f"    bigA = np.block([\n")
    f.write(f"        [A, -J.T],\n")
    f.write(f"        [J, np.zeros((2,2))]\n")
    f.write(f"    ])\n\n")
    
    f.write(f"    bigB = np.block([\n")
    f.write(f"        [b],\n")
    f.write(f"        [ np.reshape(-Jdot @ qdot.T, (2, 1)) ]\n")
    f.write(f"    ])\n\n")
    
    f.write(f"    x = np.linalg.solve(bigA, bigB)\n\n")
    
    f.write(f"    output = np.array([u1, x[0,0], u2, x[1,0], u3, x[2,0], u4, x[3,0]])\n\n")
    f.write(f"    return output\n\n")
    