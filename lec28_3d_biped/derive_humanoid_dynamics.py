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
omega_12 = phid * u.T

r = sy.Matrix([0, 0, 0])
u = sy.Matrix([0, 1, 0])
T23 = revolute(theta,r,u); #pitch
R23 = T23[0:3,0:3]
omega_23 = thetad * u.T

r = sy.Matrix([0, 0, 0])
u = sy.Matrix([0, 0, 1])
T34 = revolute(psi,r,u); #roll
omega_34 = psid * u.T

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

# print(LA[0])
# print(RA[0])

# 왜 여기 - 붙었지?
pos_hip_l_stance = (-LA).subs([(x,0), (y,0), (z,0)])
pos_hip_r_stance = (-RA).subs([(x,0), (y,0), (z,0)])


# # TODO: hip_positions file 
# # f_pos_hip = sy.lambdify([phi, theta, psi, phi_lh, theta_lh, psi_lh, phi_rh, theta_rh, psi_rh], [pos_hip_l_stance, pos_hip_r_stance], "numpy")

# ### collision detection condition ###
# print(f"gstop = {sy.simplify(LA[2]-RA[2])}")


# omega_13 = omega_12 + R12*omega_23;

# R13 = R12*R23;
# omega_14 = omega_13 + R13*omega_34;


# R14 = R13*R34;
# omega_15l = omega_14 + R14*omega_45l;
# omega_15r = omega_14 + R14*omega_45r;


# R15l = R14*R45l;
# omega_16l = omega_15l + R15l*omega_56l;
# R15r = R14*R45r;
# omega_16r = omega_15r + R15r*omega_56r;

# R16l = R15l*R56l;
# omega_17l = omega_16l + R16l*omega_67l;
# R16r = R15r*R56r;
# omega_17r = omega_16r + R16r*omega_67r;

# R17l = R16l*R67l;
# omega_18l = omega_17l + R17l*omega_78l;
# R17r = R16r*R67r;
# omega_18r = omega_17r + R17r*omega_78r;
# R18l = R17l*R78l;
# R18r = R17r*R78r;

# ### angular velocity in body frame ###
# omegaB_2 = omega_12;
# omegaB_3 = omega_23 + R23'*omegaB_2;
# omegaB_4 = omega_34 + R34'*omegaB_3;

# % A = jacobian(omegaB_4,[phid,thetad,psid]) %used in sdfast to convert
# % euler to body frame angular velocity for torso

# omegaB_5l = omega_45l + R45l'*omegaB_4;
# omegaB_6l = omega_56l + R56l'*omegaB_5l;
# omegaB_7l = omega_67l + R67l'*omegaB_6l;
# omegaB_8l = omega_78l + R78l'*omegaB_7l;

# omegaB_5r = omega_45r + R45r'*omegaB_4;
# omegaB_6r = omega_56r + R56r'*omegaB_5r;
# omegaB_7r = omega_67r + R67r'*omegaB_6r;
# omegaB_8r = omega_78r + R78r'*omegaB_7r;

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# I_LA(1) = P*(LK(1)-LA(1))/l2;
# I_LA(2) = P*(LK(2)-LA(2))/l2;
# I_LA(3) = P*(LK(3)-LA(3))/l2;

# I_RA(1) = P*(RK(1)-RA(1))/l2;
# I_RA(2) = P*(RK(2)-RA(2))/l2;
# I_RA(3) = P*(RK(3)-RA(3))/l2;
# % f_impulse = matlabFunction(I_LA, I_RA,...
# %    'File','foot_impulse','Outputs',{'I_LA', 'I_RA'});


# %%%%%% velocity and acceleration vectors %%%%%%
# q = [x y z phi theta psi phi_lh theta_lh psi_lh theta_lk phi_rh theta_rh psi_rh theta_rk];
# qdot = [xd yd zd phid thetad psid phi_lhd theta_lhd psi_lhd theta_lkd phi_rhd theta_rhd psi_rhd theta_rkd];
# qddot = [xdd ydd zdd phidd thetadd psidd phi_lhdd theta_lhdd psi_lhdd theta_lkdd phi_rhdd theta_rhdd psi_rhdd theta_rkdd];
