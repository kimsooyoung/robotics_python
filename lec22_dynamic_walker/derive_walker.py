import sympy as sy

M, I, l = sy.symbols('M I l') # Mass Hip, leg Inertia, leg length
gam, g  = sy.symbols('gam g') #Slope of ramp, gravity

x, y = sy.symbols('x y') #position of the stance leg
vx, vy = sy.symbols('vx vy') #velocity of the stance leg
ax, ay = sy.symbols('ax ay') #acceleration of the stance leg

theta1, theta2 = sy.symbols('theta1 theta2') #Angles as defined in figures
omega1 = sy.symbols('omega1') #Angular velocity
alpha1 = sy.symbols('alpha1') #Angular acceleration

theta1_n, omega1_n = sy.symbols('theta1_n omega1_n') #angle, velocity before heelstrike

pi = sy.pi
cos = sy.cos
sin = sy.sin

angle_1 = pi/2 + theta1
angle_2 = theta2 - pi

H_01 = sy.Matrix([
    [cos(angle_1), -sin(angle_1), x],
    [sin(angle_1), cos(angle_1), y],
    [0, 0, 1]
])
H_12 = sy.Matrix([
    [cos(angle_2), -sin(angle_2), l],
    [sin(angle_2), cos(angle_2), 0],
    [0, 0, 1]
])
H_02 = H_01 * H_12

C1 = sy.Matrix([x, y, 1])
H  = H_01 * sy.Matrix([l, 0, 1])
C2 = sy.simplify(H_02 * sy.Matrix([l, 0, 1]))

# print(C1)
# print(H)
# print(C2)

##############################
#####  Step 2. velocity  #####
##############################

q = sy.Matrix([x, y, theta1])
q_d = sy.Matrix([vx, vy, omega1])

H_xy = sy.Matrix([H[0], H[1]])
v_H = H_xy.jacobian(q) * q_d

##############################
##### Step 3. E-L Method #####
##############################

# 위치에너지를 위해 y값에 경사각 반영
H_og = sy.Matrix([
    [cos(-gam), -sin(-gam), 0],
    [sin(-gam), cos(-gam), 0],
    [0, 0, 1]
])
R_H = H_og * H

# swing leg dynamics are neglected so that
# the foot placement angle can be set instantaneously fast.
T = 0.5 * M * v_H.dot(v_H) + \
    0.5 * I * omega1**2
V = sy.simplify(M * g * R_H[1])
L = T - V

dL_dq_d = []
dt_dL_dq_d = []
dL_dq = []
q_dd = sy.Matrix([ax, ay, alpha1])

EOM = []

for i in range(len(q_dd)):
    dL_dq_d.append(sy.diff(L, q_d[i]))
    temp = 0
    for j in range(len(q_dd)):
        temp += sy.diff(dL_dq_d[i], q[j]) * q_d[j] + \
                sy.diff(dL_dq_d[i], q_d[j]) * q_dd[j]
    dt_dL_dq_d.append(temp)
    dL_dq.append(sy.diff(L, q[i]))
    EOM.append(dt_dL_dq_d[i] - dL_dq[i])

EOM = sy.Matrix(EOM)

# print(EOM[0])
# print(EOM[1])
# print(EOM[2])

# Ax = b
A_ss = EOM.jacobian(q_dd)
b_ss = []

for i in range(len(q_dd)):
    b_ss.append(-1 * EOM[i].subs([(ax, 0), (ay, 0), (alpha1, 0) ]))
    
# We only use the elements from alpha1
print(f"A_ss = {sy.simplify(A_ss[2,2])}")
print(f"b_ss = {sy.simplify(b_ss[2])}")

# reaction forces
Rx = sy.simplify(EOM[0].subs([(ax, 0), (ay, 0)]))
Ry = sy.simplify(EOM[1].subs([(ax, 0), (ay, 0)]))
print(f"Rx = {Rx}")
print(f"Ry = {Ry}")

C2_xy = sy.Matrix([ C2[0], C2[1] ])
J_C2 = C2_xy.jacobian(q)

A_n_hs = A_ss.subs([ (theta1, theta1_n)])
J_n_sw = J_C2.subs([ (theta1, theta1_n)])

# hs equations
print('J11 = ', sy.simplify(J_n_sw[0,0]))
print('J12 = ', sy.simplify(J_n_sw[0,1]))
print('J13 = ', sy.simplify(J_n_sw[0,2]))
print('J21 = ', sy.simplify(J_n_sw[1,0]))
print('J22 = ', sy.simplify(J_n_sw[1,1]))
print('J23 = ', sy.simplify(J_n_sw[1,2]),'\n')
print('J = [J11 J12 J13; J21 J22 J23]');

print('A11 = ', sy.simplify(A_n_hs[0,0]))
print('A12 = ', sy.simplify(A_n_hs[0,1]))
print('A13 = ', sy.simplify(A_n_hs[0,2]),'\n')

print('A21 = ', sy.simplify(A_n_hs[1,0]))
print('A22 = ', sy.simplify(A_n_hs[1,1]))
print('A23 = ', sy.simplify(A_n_hs[1,2]),'\n')

print('A31 = ', sy.simplify(A_n_hs[2,0]))
print('A32 = ', sy.simplify(A_n_hs[2,1]))
print('A33 = ', sy.simplify(A_n_hs[2,2]),'\n')
print('A_n_hs = [A11 A12 A13; A21 A22 A23; A31 A32 A33]');

print('X_n_hs = [0, 0, omega1_n]')
print('b_hs = [A_n_hs*X_n_hs; 0; 0]')
print('A_hs = [A_n_hs, -J'' ; J, zeros(2,2)]')
print('X_hs = A_hs\b_hs')
print('omega1 = X_hs[2]')
