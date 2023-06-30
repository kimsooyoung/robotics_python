import sympy as sy


def revolute(theta, r, u):
    
    ux, uy, uz = u
    
    cth = sy.cos(theta)
    sth = sy.sin(theta)
    vth = 1 - cth
    
    R11 = ux**2 * vth + cth;  R12 = ux*uy*vth - uz*sth; R13 = ux*uz*vth + uy*sth
    R21 = ux*uy*vth + uz*sth; R22 = uy**2*vth + cth   ; R23 = uy*uz*vth - ux*sth
    R31 = ux*uz*vth - uy*sth; R32 = uy*uz*vth + ux*sth; R33 = uz**2*vth + cth
    
    R = sy.Matrix([
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33]
    ])

    I = sy.eye(3)
    
    T12 = (I-R)*r
    
    T = sy.Matrix([
        [R11, R12, R13, T12[0]],
        [R21, R22, R23, T12[1]],
        [R31, R32, R33, T12[2]],
        [0,   0,   0,   1]
    ])
    
    return T

# position, angle, leg angular velocity
x, y, z = sy.symbols('x y z')
phi, theta, psi = sy.symbols('phi theta psi')
phi_lh, theta_lh, psi_lh, theta_lk = sy.symbols('phi_lh theta_lh psi_lh theta_lk')
phi_rh, theta_rh, psi_rh, theta_rk = sy.symbols('phi_rh theta_rh psi_rh theta_rk')

# velocity, angular velocity, leg angular velocity
xd, yd, zd = sy.symbols('xd yd zd')
phid, thetad, psid = sy.symbols('phid thetad psid')
phi_lhd, theta_lhd, psi_lhd, theta_lkd = sy.symbols('phi_lhd theta_lhd psi_lhd theta_lkd')
phi_rhd, theta_rhd, psi_rhd, theta_rkd = sy.symbols('phi_rhd theta_rhd psi_rhd theta_rkd')

# acceleration, angular acceleration, leg angular acceleration
xdd, ydd, zdd = sy.symbols('xdd ydd zdd')
phidd, thetadd, psidd = sy.symbols('phidd thetadd psidd')
phi_lhdd, theta_lhdd, psi_lhdd, theta_lkdd = sy.symbols('phi_lhdd theta_lhdd psi_lhdd theta_lkdd')
phi_rhdd, theta_rhdd, psi_rhdd, theta_rkdd = sy.symbols('phi_rhdd theta_rhdd psi_rhdd theta_rkdd')

# link length
w, l0, l1, l2 = sy.symbols('w l0 l1 l2')

# Forces
g, P = sy.symbols('g P')

# Inertia, Mass
Ibx, Iby, Ibz = sy.symbols('Ibx Iby Ibz')
Itx, Ity, Itz = sy.symbols('Itx Ity Itz')
Icx, Icy, Icz = sy.symbols('Icx Icy Icz')
mb, mt, mc = sy.symbols('mb mt mc')

dof = 14

### Position Vectors => We'll gonna use zero-ref kinematics ###

### world frame ###
T01 = sy.Matrix([
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1]
])
R01 = T01[0:3,0:3]

r = sy.Matrix([0, 0, 0])
u = sy.Matrix([1, 0, 0])
T12 = revolute(phi,r,u); #yaw
R12 = T12[0:3,0:3]
omega_12 = phid * u
# TODO: check transpose requiured or not 

r = sy.Matrix([0, 0, 0])
u = sy.Matrix([0, 1, 0])
T23 = revolute(theta,r,u); #pitch
R23 = T23[0:3,0:3]
omega_23 = thetad * u

r = sy.Matrix([0, 0, 0])
u = sy.Matrix([0, 0, 1])
T34 = revolute(psi,r,u); #roll
R34 = T34[0:3,0:3]
omega_34 = psid * u

### left side ###
r = sy.Matrix([0, w, 0])
u = sy.Matrix([0, 0, 1])
T45l = revolute(psi_lh,r,u); #roll
R45l = T45l[0:3,0:3]
omega_45l = psi_lhd * u

r = sy.Matrix([0, w, 0])
u = sy.Matrix([1, 0, 0])
T56l = revolute(phi_lh,r,u); #yaw
R56l = T56l[0:3,0:3]
omega_56l = phi_lhd * u

r = sy.Matrix([0, w, 0])
u = sy.Matrix([0, -1, 0])
T67l = revolute(theta_lh,r,u); #pitch
R67l = T67l[0:3,0:3]
omega_67l = theta_lhd * u

r = sy.Matrix([0, w, -l1])
u = sy.Matrix([0, -1, 0])
T78l = revolute(theta_lk,r,u); #pitch
R78l = T78l[0:3,0:3]
omega_78l = theta_lkd * u


### right side ###
r = sy.Matrix([0, -w, 0])
u = sy.Matrix([0, 0, -1])
T45r = revolute(psi_rh,r,u); #roll
R45r = T45r[0:3,0:3]
omega_45r = psi_rhd * u

r = sy.Matrix([0, -w, 0])
u = sy.Matrix([-1, 0, 0])
T56r = revolute(phi_rh,r,u); #yaw
R56r = T56r[0:3,0:3]
omega_56r = phi_rhd * u

r = sy.Matrix([0, -w, 0])
u = sy.Matrix([0, -1, 0])
T67r = revolute(theta_rh,r,u); #pitch
R67r = T67r[0:3,0:3]
omega_67r = theta_rhd * u

r = sy.Matrix([0, -w, -l1])
u = sy.Matrix([0, -1, 0])
T78r = revolute(theta_rk,r,u); #pitch
R78r = T78r[0:3,0:3]
omega_78r = theta_rkd * u


### position vectors for all joints ###
B = T01[0:3,3] #base

T04 = T01*T12*T23*T34;
H = sy.simplify(T04 * sy.Matrix([0, 0, l0, 1]))

T07l = T01*T12*T23*T34*T45l*T56l*T67l;
LH = sy.simplify(T07l * sy.Matrix([0, w, 0, 1]))

T08l = T07l*T78l;
LK = sy.simplify(T08l * sy.Matrix([0, w, -l1, 1]))

# LA = sy.simplify(T08l * sy.Matrix([0, w, -(l1+l2), 1]))
LA = (T08l * sy.Matrix([0, w, -(l1+l2), 1]))

T07r = T01*T12*T23*T34*T45r*T56r*T67r
RH = sy.simplify(T07r * sy.Matrix([0, -w, 0, 1]))

T08r = T07r*T78r
RK = sy.simplify(T08r * sy.Matrix([0, -w, -l1, 1]))

# RA = sy.simplify(T08r * sy.Matrix([0, -w, -(l1+l2), 1]))
RA = (T08r * sy.Matrix([0, -w, -(l1+l2), 1]))

### all center of mass ###
b = sy.simplify(T04*sy.Matrix([0, 0, 0.5*l0, 1]))
lt = sy.simplify(T07l*sy.Matrix([0, w, -0.5*l1, 1]))
lc = sy.simplify(T08l*sy.Matrix([0, w, -(l1+0.5*l2), 1]))
rt = sy.simplify(T07r*sy.Matrix([0, -w, -0.5*l1, 1]))
rc = sy.simplify(T08r*sy.Matrix([0, -w, -(l1+0.5*l2), 1]))

print(f"[     ] Position Vector Caculation Done")
# print(LA[0])
# print(RA[0])

# 왜 여기 - 붙었지?
# pos_hip_l_stance = (-LA).subs([(x,0), (y,0), (z,0)])
# pos_hip_r_stance = (-RA).subs([(x,0), (y,0), (z,0)])

# print(pos_hip_l_stance)
# print()
# print(pos_hip_r_stance)

### collision detection condition ###
# print(f"gstop = {sy.simplify(LA[2]-RA[2])}")

omega_13 = omega_12 + R12*omega_23

R13 = R12 * R23
omega_14 = omega_13 + R13*omega_34

R14 = R13 * R34
omega_15l = omega_14 + R14*omega_45l
omega_15r = omega_14 + R14*omega_45r

R15l = R14*R45l
omega_16l = omega_15l + R15l*omega_56l
R15r = R14*R45r
omega_16r = omega_15r + R15r*omega_56r

R16l = R15l*R56l
omega_17l = omega_16l + R16l*omega_67l
R16r = R15r*R56r
omega_17r = omega_16r + R16r*omega_67r

R17l = R16l*R67l
omega_18l = omega_17l + R17l*omega_78l
R17r = R16r*R67r
omega_18r = omega_17r + R17r*omega_78r
R18l = R17l*R78l
R18r = R17r*R78r

### angular velocity in body frame ###
omegaB_2 = omega_12
omegaB_3 = omega_23 + R23.T * omegaB_2
omegaB_4 = omega_34 + R34.T * omegaB_3

# % A = jacobian(omegaB_4,[phid,thetad,psid]) %used in sdfast to convert
# % euler to body frame angular velocity for torso

omegaB_5l = omega_45l + R45l.T * omegaB_4
omegaB_6l = omega_56l + R56l.T * omegaB_5l
omegaB_7l = omega_67l + R67l.T * omegaB_6l
omegaB_8l = omega_78l + R78l.T * omegaB_7l

omegaB_5r = omega_45r + R45r.T * omegaB_4
omegaB_6r = omega_56r + R56r.T * omegaB_5r
omegaB_7r = omega_67r + R67r.T * omegaB_6r
omegaB_8r = omega_78r + R78r.T * omegaB_7r

print(f"[=    ] Angular Velocity Caculation Done")

### mass Intertia ###

I_LA_1 = P * (LK[0] - LA[0]) / l2
I_LA_2 = P * (LK[1] - LA[1]) / l2
I_LA_3 = P * (LK[2] - LA[2]) / l2

I_RA_1 = P * (RK[0] - RA[0]) / l2
I_RA_2 = P * (RK[1] - RA[1]) / l2
I_RA_3 = P * (RK[2] - RA[2]) / l2

# % f_impulse = matlabFunction(I_LA, I_RA,...
# %    'File','foot_impulse','Outputs',{'I_LA', 'I_RA'});

### velocity and acceleration vectors ###

q = sy.Matrix([x, y, z, phi, theta, psi, phi_lh, theta_lh, psi_lh, theta_lk, phi_rh, theta_rh, psi_rh, theta_rk])
qdot = sy.Matrix([xd, yd, zd, phid, thetad, psid, phi_lhd, theta_lhd, psi_lhd, theta_lkd, phi_rhd, theta_rhd, psi_rhd, theta_rkd])
qddot = sy.Matrix([xdd, ydd, zdd, phidd, thetadd, psidd, phi_lhdd, theta_lhdd, psi_lhdd, theta_lkdd, phi_rhdd, theta_rhdd, psi_rhdd, theta_rkdd])

v_b = b.jacobian(q) * qdot
v_rt = rt.jacobian(q) * qdot
v_rc = rc.jacobian(q) * qdot
v_lt = lt.jacobian(q) * qdot
v_lc = lc.jacobian(q) * qdot
v_RA = RA.jacobian(q) * qdot
v_LA = LA.jacobian(q) * qdot

print(f"[==   ] Linear Velocity Caculation Done")

# vel_hip_l_stance = (-v_LA).subs([(x,0), (y,0), (z,0)])
# vel_hip_r_stance = (-v_RA).subs([(x,0), (y,0), (z,0)])

# print(vel_hip_l_stance)
# print()
# print(vel_hip_r_stance)

### Potential, Kinetic, and Total Energy ###

Ib = sy.Matrix([
    [Ibx, 0, 0],
    [0, Iby, 0],
    [0, 0, Ibz]
])

It = sy.Matrix([
    [Itx, 0, 0],
    [0, Ity, 0],
    [0, 0, Itz]
])

Ic = sy.Matrix([
    [Icx, 0, 0],
    [0, Icy, 0],
    [0, 0, Icz]
])

# print(v_b)
# print(v_rt.dot(v_rt))
# print(omegaB_4.dot(omegaB_4))

# T = 0.5 * mb * v_b.dot(v_b)
T = 0.5 * mb * v_b.dot(v_b) + \
    0.5 * mt * v_rt.dot(v_rt) + \
    0.5 * mc * v_rc.dot(v_rc) + \
    0.5 * mt * v_lt.dot(v_lt) + \
    0.5 * mc * v_lc.dot(v_lc) + \
    0.5 * omegaB_4.dot(Ib * omegaB_4) + \
    0.5 * omegaB_7l.dot(It * omegaB_7l) + \
    0.5 * omegaB_7r.dot(It * omegaB_7r) + \
    0.5 * omegaB_8l.dot(Ic * omegaB_8l) + \
    0.5 * omegaB_8r.dot(Ic * omegaB_8r)

V = mb * g * b[2] + \
    mt * g * rt[2] + \
    mt * g * lt[2] + \
    mc * g * lc[2] + \
    mc * g * rc[2]
    
L = T - V

print(f"[===  ] Energy Caculation Done")

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

print(f"[==== ] EOM Caculation Done")

# q = sy.Matrix([
    # x, y, z, phi, theta, psi, 
    # phi_lh, theta_lh, psi_lh, theta_lk, 
    # phi_rh, theta_rh, psi_rh, theta_rk])
EOM = sy.Matrix([
    EOM[0],EOM[1],EOM[2], EOM[3],EOM[4],EOM[5],
    EOM[6],EOM[7],EOM[8],EOM[9],
    EOM[10],EOM[11],EOM[12],EOM[13]
])

M = EOM.jacobian(qddot)
b = EOM.subs([
    *list(zip(qddot, [0]*len(qddot))),
])
G = b.subs([
    *list(zip(qdot, [0]*len(qdot))),
])

C = b - G

### jacobians and its rate ###
p_right_ankle = sy.Matrix(RA[0:3])
p_left_anke = sy.Matrix(LA[0:3])

J_l = p_right_ankle.jacobian(q)
J_r = p_left_anke.jacobian(q)
col, row = J_l.shape

Jdot_l = []
Jdot_r = []

for i in range(col):
    J_temp_l = J_l[i,:].jacobian(q) * qdot
    J_temp_r = J_r[i,:].jacobian(q) * qdot
    Jdot_l.append(list(J_temp_l))
    Jdot_r.append(list(J_temp_r))

print(f"[=====] M C G J Jdot Calculation Done")

print("Generating rhs file")

dof = len(q)

with open("humanoid_rhs.py", "w") as f:
    
    f.write("import numpy as np \n\n")
    f.write("def cos(angle): \n")
    f.write("    return np.cos(angle) \n\n")
    f.write("def sin(angle): \n")
    f.write("    return np.sin(angle) \n\n")

    f.write("def humanoid_rhs(z, t, params): \n\n")
    
    f.write("    x, xd, y, yd, z, zd, phi, phid, theta, thetad, psi, psid, \ \n")
    f.write("    phi_lh, phi_lhd, theta_lh, theta_lhd, \ \n")
    f.write("    psi_lh, psi_lhd, theta_lk, theta_lkd, \ \n")
    f.write("    phi_rh, phi_rhd, theta_rh, theta_rhd, \ \n")
    f.write("    psi_rh, psi_rhd, theta_rk, theta_rkd = z \n\n")
    
    f.write("    mb, mt, mc = params.mb, params.mt, params.mc \n")
    f.write("    Ibx, Iby, Ibz = params.Ibx, params.Iby, params.Ibz \n")
    f.write("    Itx, Ity, Itz = params.Itx, params.Ity, params.Itz \n")
    f.write("    Icx, Icy, Icz = params.Icx, params.Icy, params.Icz \n")
    f.write("    l0, l1, l2 = params.l0, params.l1, params.l2 \n")
    f.write("    w, g = params.w, params.g \n\n")
    
    for i in range(dof):
        for j in range(dof):
            elem = sy.simplify(M[i,j])
            f.write(f"    M{i+1}{j+1} = {elem} \n\n")
    f.write("\n")
    
    for i in range(dof):
        elem = sy.simplify( C[i] )
        f.write(f"    C{i+1} = {elem} \n\n")
    f.write("\n")
    
    for i in range(dof):
        elem = sy.simplify( G[i] )
        f.write(f"    G{i+1} = {elem} \n\n")
    f.write("\n")
    
    for i in range(3):
        for j in range(dof):
            elem = sy.simplify(J_l[i,j])
            f.write(f"    J_l{i+1}{j+1} = {elem} \n\n")
    f.write("\n")
    
    for i in range(3):
        for j in range(dof):
            elem = sy.simplify(J_r[i,j])
            f.write(f"    J_r{i+1}{j+1} = {elem} \n\n")
    f.write("\n")
    
    for i in range(3):
        for j in range(dof):
            elem = sy.simplify(Jdot_l[i,j])
            f.write(f"    Jdot_l{i+1}{j+1} = {elem} \n\n")
    f.write("\n")
    
    for i in range(3):
        for j in range(dof):
            elem = sy.simplify(Jdot_r[i,j])
            f.write(f"    Jdot_l{i+1}{j+1} = {elem} \n\n")
    f.write("\n")
    
    
    f.write("    M = np.array([")
    for i in range(dof):
        f.write(f"        [")
        for j in range(dof):
            f.write(f"M{i+1}{j+1}")
            if j != dof-1:
                f.write(f", ")
        f.write(f"]")
        if i != dof-1:
            f.write(f",\n")
        else:
            f.write(f"\n")
    f.write("    ]) \n\n")
    
    f.write("    b = -np.array([ \n")
    for i in range(dof):
        f.write(f"        [C{i+1} + G{i+1}]")
        if i != dof-1:
            f.write(f",")
        f.write(f"\n")
    f.write("    ]) \n\n")
    
    f.write("    J_l = np.array([ \n")
    for i in range(dof):
        f.write(f"        [")
        for j in range(3):
            f.write(f"J_l{i+1}{j+1}")
            if j != dof-1:
                f.write(f", ")
        f.write(f"]")
        if i != dof-1:
            f.write(f",\n")
        else:
            f.write(f"\n")
    f.write("    ]) \n\n")
    
    f.write("    J_r = np.array([ \n")
    for i in range(3):
        f.write(f"        [")
        for j in range(dof):
            f.write(f"J_r{i+1}{j+1}")
            if j != dof-1:
                f.write(f", ")
        f.write(f"]")
        if i != dof-1:
            f.write(f",\n")
        else:
            f.write(f"\n")
    f.write("    ]) \n\n")
    
    f.write("    Jdot_l = np.array([ \n")
    for i in range(3):
        f.write(f"        [")
        for j in range(dof):
            f.write(f"Jdot_l{i+1}{j+1}")
            if j != dof-1:
                f.write(f", ")
        f.write(f"]")
        if i != dof-1:
            f.write(f",\n")
        else:
            f.write(f"\n")
    f.write("    ]) \n\n")
    
    f.write("    Jdot_r = np.array([ \n")
    for i in range(3):
        f.write(f"        [")
        for j in range(dof):
            f.write(f"Jdot_r{i+1}{j+1}")
            if j != dof-1:
                f.write(f", ")
        f.write(f"]")
        if i != dof-1:
            f.write(f",\n")
        else:
            f.write(f"\n")
    f.write("    ]) \n\n")
    
    print("Generating humanoid_rhs done...")

# with open("single_stance.py", "w") as f:
    
#     f.write("import numpy as np \n\n")
#     f.write("def cos(angle): \n")
#     f.write("    return np.cos(angle) \n\n")
#     f.write("def sin(angle): \n")
#     f.write("    return np.sin(angle) \n\n")

#     f.write("def single_stance(z, t, params): \n\n")
    
    