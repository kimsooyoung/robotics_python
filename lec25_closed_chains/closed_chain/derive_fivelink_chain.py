import sympy as sy

q1, q2, q3, q4, q5 = sy.symbols('q1 q2 q3 q4 q5') # Angles as defined in figures
u1, u2, u3, u4, u5 = sy.symbols('u1 u2 u3 u4 u5') # Angular velocity
a1, a2, a3, a4, a5 = sy.symbols('a1 a2 a3 a4 a5') # Angular acceleration
m, I, g, c, l = sy.symbols('m I g c l') # System parameters

c = l / 2

m1, m2, m3, m4, m5 = m, m, m, m, m # Each link mass
I1, I2, I3, I4, I5 = I, I, I, I, I # Each link inertia
c1, c2, c3, c4, c5 = c, c, c, c, c # COM Distance from joint

q = sy.Matrix([q1, q2, q3, q4, q5])
qdot = sy.Matrix([u1, u2, u3, u4, u5])
qddot = sy.Matrix([a1, a2, a3, a4, a5])

### derive positions ###
mpi = sy.pi
sin1 = sy.sin(q1 + 3*mpi/2); cos1 = sy.cos(q1 + 3*mpi/2)
sin2 = sy.sin(q2); cos2 = sy.cos(q2)
sin3 = sy.sin(q3); cos3 = sy.cos(q3)
sin4 = sy.sin(q4); cos4 = sy.cos(q4)
sin5 = sy.sin(q5); cos5 = sy.cos(q5)

H01 = sy.Matrix([ [cos1, -sin1, 0],
                  [sin1, cos1,  0],
                  [0,   0,  1] ])
H12 = sy.Matrix([ [cos2, -sin2, l],
                  [sin2, cos2,  0],
                  [0,   0,  1] ])
H23 = sy.Matrix([ [cos3, -sin3, l],
                  [sin3, cos3,  0],
                  [0,   0,  1] ])
H34 = sy.Matrix([ [cos4, -sin4, l],
                  [sin4, cos4,  0],
                  [0,   0,  1] ])
H45 = sy.Matrix([ [cos5, -sin5, l],
                  [sin5, cos5,  0],
                  [0,   0,  1] ])

H02 = H01 * H12; H03 = H02 * H23; 
H04 = H03 * H34; H05 = H04 * H45;

G1 = H01 * sy.Matrix([c1, 0, 1])
G2 = H02 * sy.Matrix([c2, 0, 1])
G3 = H03 * sy.Matrix([c3, 0, 1])
G4 = H04 * sy.Matrix([c4, 0, 1])
G5 = H05 * sy.Matrix([c5, 0, 1])

G1_xy = sy.Matrix([G1[0], G1[1]])
G2_xy = sy.Matrix([G2[0], G2[1]])
G3_xy = sy.Matrix([G3[0], G3[1]])
G4_xy = sy.Matrix([G4[0], G4[1]])
G5_xy = sy.Matrix([G5[0], G5[1]])

### velocity vectors ###
v_G1 = G1_xy.jacobian(q) * qdot
v_G2 = G2_xy.jacobian(q) * qdot
v_G3 = G3_xy.jacobian(q) * qdot
v_G4 = G4_xy.jacobian(q) * qdot
v_G5 = G5_xy.jacobian(q) * qdot

### end of last link is P (say) then position and velocity is ###
P = H05 * sy.Matrix([l, 0, 1])
P_xy = sy.Matrix([
    sy.simplify(P[0]),
    sy.simplify(P[1])
])
# print("x_P = ", P_xy[0])
# print("y_P = ", P_xy[1])

v_P = P_xy.jacobian(q) * qdot
# print("vx_P = ", v_P[0])
# print("vy_P = ", v_P[1])

om1 = u1;
om2 = om1 + u2;
om3 = om2 + u3;
om4 = om3 + u4;
om5 = om4 + u5;

### lagrangian ###
T = 0.5*m1*v_G1.dot(v_G1) + \
    0.5*m2*v_G2.dot(v_G2) + \
    0.5*m3*v_G3.dot(v_G3) + \
    0.5*m4*v_G4.dot(v_G4) + \
    0.5*m5*v_G5.dot(v_G5) + \
    0.5*I1*om1**2 + \
    0.5*I2*om2**2 + \
    0.5*I3*om3**2 + \
    0.5*I4*om4**2 + \
    0.5*I5*om5**2

V = m1*g*G1[1] + m2*g*G2[1] + m3*g*G3[1] + m4*g*G4[1] + m5*g*G5[1]
L = T - V

### Derive EL equation ###

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
    
    # 현재 외력이 0이므로 이 두개 항만 있다.
    EOM.append(dt_dL_dq_d[i] - dL_dq[i])

EOM = sy.Matrix([EOM[0],EOM[1],EOM[2],EOM[3],EOM[4]])

### collecting equations as A a = b ###

M = EOM.jacobian(qddot)
b = EOM.subs([
    (a1, 0),
    (a2, 0),
    (a3, 0),
    (a4, 0),
    (a5, 0)
])
G = b.subs([
    (u1, 0),
    (u2, 0),
    (u3, 0),
    (u4, 0),
    (u5, 0)
])
C = b - G

J = P_xy.jacobian(q)
col, row = J.shape
Jdot = []

for i in range(col):
    J_temp = J[i,:].jacobian(q) * qdot
    Jdot.append(list(J_temp))

with open("fivelinkchain_rhs.py", "w") as f:
    
    f.write("import numpy as np \n\n")
    f.write("def cos(angle): \n")
    f.write("    return np.cos(angle) \n\n")
    f.write("def sin(angle): \n")
    f.write("    return np.sin(angle) \n\n")
    
    f.write("def five_link_chain(z0, t, params): \n\n")
    f.write("    q1, u1 = z0[0], z0[1] \n")
    f.write("    q2, u2 = z0[2], z0[3] \n")
    f.write("    q3, u3 = z0[4], z0[5] \n")
    f.write("    q4, u4 = z0[6], z0[7] \n")
    f.write("    q5, u5 = z0[8], z0[9] \n\n")
    f.write("    m, I, g, l = params.m, params.I, params.g, params.l \n\n")
    
    f.write(f"    M11 = {sy.simplify(M[0,0])} \n\n")
    f.write(f"    M12 = {sy.simplify(M[0,1])} \n\n")
    f.write(f"    M13 = {sy.simplify(M[0,2])} \n\n")
    f.write(f"    M14 = {sy.simplify(M[0,3])} \n\n")
    f.write(f"    M15 = {sy.simplify(M[0,4])} \n\n")
    
    f.write(f"    M21 = {sy.simplify(M[1,0])} \n\n")
    f.write(f"    M22 = {sy.simplify(M[1,1])} \n\n")
    f.write(f"    M23 = {sy.simplify(M[1,2])} \n\n")
    f.write(f"    M24 = {sy.simplify(M[1,3])} \n\n")
    f.write(f"    M25 = {sy.simplify(M[1,4])} \n\n")
    
    f.write(f"    M31 = {sy.simplify(M[2,0])} \n\n")
    f.write(f"    M32 = {sy.simplify(M[2,1])} \n\n")
    f.write(f"    M33 = {sy.simplify(M[2,2])} \n\n")
    f.write(f"    M34 = {sy.simplify(M[2,3])} \n\n")
    f.write(f"    M35 = {sy.simplify(M[2,4])} \n\n")
    
    f.write(f"    M41 = {sy.simplify(M[3,0])} \n\n")
    f.write(f"    M42 = {sy.simplify(M[3,1])} \n\n")
    f.write(f"    M43 = {sy.simplify(M[3,2])} \n\n")
    f.write(f"    M44 = {sy.simplify(M[3,3])} \n\n")
    f.write(f"    M45 = {sy.simplify(M[3,4])} \n\n")
    
    f.write(f"    M51 = {sy.simplify(M[4,0])} \n\n")
    f.write(f"    M52 = {sy.simplify(M[4,1])} \n\n")
    f.write(f"    M53 = {sy.simplify(M[4,2])} \n\n")
    f.write(f"    M54 = {sy.simplify(M[4,3])} \n\n")
    f.write(f"    M55 = {sy.simplify(M[4,4])} \n\n")
    
    f.write(f"    C1 = {C[0]}\n\n")
    f.write(f"    C2 = {C[1]}\n\n")
    f.write(f"    C3 = {C[2]}\n\n")
    f.write(f"    C4 = {C[3]}\n\n")
    f.write(f"    C5 = {C[4]}\n\n")
    
    f.write(f"    G1 = {G[0]}\n\n")
    f.write(f"    G2 = {G[1]}\n\n")
    f.write(f"    G3 = {G[2]}\n\n")
    f.write(f"    G4 = {G[3]}\n\n")
    f.write(f"    G5 = {G[4]}\n\n")
    
    f.write(f"    J11 = {J[0,0]}\n\n")
    f.write(f"    J12 = {J[0,1]}\n\n")
    f.write(f"    J13 = {J[0,2]}\n\n")
    f.write(f"    J14 = {J[0,3]}\n\n")
    f.write(f"    J15 = {J[0,4]}\n\n")
    f.write(f"    J21 = {J[1,0]}\n\n")
    f.write(f"    J22 = {J[1,1]}\n\n")
    f.write(f"    J23 = {J[1,2]}\n\n")
    f.write(f"    J24 = {J[1,3]}\n\n")
    f.write(f"    J25 = {J[1,4]}\n\n")
    
    f.write(f"    Jdot11 = {Jdot[0][0]}\n\n")
    f.write(f"    Jdot12 = {Jdot[0][1]}\n\n")
    f.write(f"    Jdot13 = {Jdot[0][2]}\n\n")
    f.write(f"    Jdot14 = {Jdot[0][3]}\n\n")
    f.write(f"    Jdot15 = {Jdot[0][4]}\n\n")
    f.write(f"    Jdot21 = {Jdot[1][0]}\n\n")
    f.write(f"    Jdot22 = {Jdot[1][1]}\n\n")
    f.write(f"    Jdot23 = {Jdot[1][2]}\n\n")
    f.write(f"    Jdot24 = {Jdot[1][3]}\n\n")
    f.write(f"    Jdot25 = {Jdot[1][4]}\n\n")
    
    f.write(f"    A = np.array([\n")
    f.write(f"        [M11, M12, M13, M14, M15],\n")
    f.write(f"        [M21, M22, M23, M24, M25],\n")
    f.write(f"        [M31, M32, M33, M34, M35],\n")
    f.write(f"        [M41, M42, M43, M44, M45],\n")
    f.write(f"        [M51, M52, M53, M54, M55]\n")
    f.write(f"        ])\n\n")
    
    f.write(f"    b = -np.array([\n")
    f.write(f"        [C1 + G1], \n")
    f.write(f"        [C2 + G2],\n")
    f.write(f"        [C3 + G3],\n")
    f.write(f"        [C4 + G4],\n")
    f.write(f"        [C5 + G5]])\n\n")
    
    f.write(f"    J = np.array([\n")
    f.write(f"        [J11, J12, J13, J14, J15],\n")
    f.write(f"        [J21, J22, J23, J24, J25],\n")
    f.write(f"    ])\n\n")
    
    f.write(f"    Jdot = np.array([\n")
    f.write(f"        [Jdot11, Jdot12, Jdot13, Jdot14, Jdot15],\n")
    f.write(f"        [Jdot21, Jdot22, Jdot23, Jdot24, Jdot25],\n")
    f.write(f"    ])\n\n")
    
    f.write(f"    qdot = np.array([u1, u2, u3, u4, u5])\n\n")
    
    f.write(f"    bigA = np.block([\n")
    f.write(f"        [A, -J.T],\n")
    f.write(f"        [J, np.zeros((2,2))]\n")
    f.write(f"    ])\n\n")
    
    f.write(f"    bigB = np.block([\n")
    f.write(f"        [b],\n")
    f.write(f"        [ np.reshape(-Jdot @ qdot.T, (2, 1)) ]\n")
    f.write(f"    ])\n\n")
    
    f.write(f"    x = np.linalg.solve(bigA, bigB)\n\n")
    
    f.write(f"    output = np.array([u1, x[0,0], u2, x[1,0], u3, x[2,0], u4, x[3,0], u5, x[4,0]])\n\n")
    f.write(f"    return output\n\n")
    
print("Done!")