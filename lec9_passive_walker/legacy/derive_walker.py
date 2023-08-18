# Copyright 2022 @RoadBalance
# Reference from https://pab47.github.io/legs.html
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sympy as sy

# define symbolic quantities
# 각도, 각속도, 각가속도
theta1, theta2 = sy.symbols('theta1 theta2', real=True)
omega1, omega2 = sy.symbols('omega1 omega2', real=True)
alpha1, alpha2 = sy.symbols('alpha1 alpha2', real=True)

theta1_n, theta2_n = sy.symbols('theta1_n theta2_n', real=True)
omega1_n, omega2_n = sy.symbols('omega1_n omega2_n', real=True)

# 변위, 속도, 가속도
x, y = sy.symbols('x y', real=True)
vx, vy = sy.symbols('vx vy', real=True)
ax, ay = sy.symbols('ax ay', real=True)

# m, M : leg mass, body mass
# I : body moment of inertia
# g : gravity
# c, l : leg length, body length
# gam : slope angle
m, M, I, g = sy.symbols('m M I g', real=True)
c, l, gam = sy.symbols('c l gam', real=True)

# 1a) position vectors
mpi = sy.pi
cos1 = sy.cos(mpi/2 + theta1)
sin1 = sy.sin(mpi/2 + theta1)
H01 = sy.Matrix([
    [cos1, -sin1, x],
    [sin1, cos1,  y],
    [0,     0,     1]
])

cos2 = sy.cos(-mpi + theta2)
sin2 = sy.sin(-mpi + theta2)
H12 = sy.Matrix([
    [cos2, -sin2, l],
    [sin2,  cos2,  0],
    [0,     0,     1]
])
H02 = H01*H12

C1 = sy.Matrix([x, y, 1])

H = H01*sy.Matrix([l, 0, 1])
# Jacobian 계산을 위해 matrix로 바꾼다.
x_H = sy.Matrix([H[0]])
y_H = sy.Matrix([H[1]])

G1 = H01*sy.Matrix([l-c, 0, 1])
x_G1 = sy.Matrix([G1[0]])
y_G1 = sy.Matrix([G1[1]])

G2 = H02*sy.Matrix([c, 0, 1])
x_G2 = sy.Matrix([G2[0]])
y_G2 = sy.Matrix([G2[1]])

C2 = H02*sy.Matrix([l, 0, 1])
x_C2 = sy.Matrix([C2[0]])
y_C2 = sy.Matrix([C2[1]])

print(f'H: {H}')
print(f'G1: {G1}')
print(f'G2: {G2}')
print(f'C2: {C2}')

# 1b) velocity vectors
q = sy.Matrix([x, y, theta1, theta2])
qdot = sy.Matrix([vx, vy, omega1, omega2])

v_H_x = x_H.jacobian(q)*qdot
v_H_y = y_H.jacobian(q)*qdot
v_G1_x = x_G1.jacobian(q)*qdot
v_G1_y = y_G1.jacobian(q)*qdot
v_G2_x = x_G2.jacobian(q)*qdot
v_G2_y = y_G2.jacobian(q)*qdot

# 2 * 1
v_H = sy.Matrix([v_H_x, v_H_y])
v_G1 = sy.Matrix([v_G1_x, v_G1_y])
v_G2 = sy.Matrix([v_G2_x, v_G2_y])

print()
print(f'v_H: {v_H}')
print(f'v_G1: {v_G1}')
print(f'v_G2: {v_G2}')

# 경사각 반영
cosGam = sy.cos(-gam)
sinGam = sy.sin(-gam)
R = sy.Matrix([
    [cosGam, -sinGam],
    [sinGam, cosGam]
])
# print(R)

# 2차원이므로 H를 그대로 쓸 수는 없다.
R_H = R*sy.Matrix([x_H, y_H])
R_G1 = R*sy.Matrix([x_G1, y_G1])
R_G2 = R*sy.Matrix([x_G2, y_G2])

# Y_H = sy.Matrix([R_H[1]]);
# Y_G1 = sy.Matrix([R_G1[1]]);
# Y_G2 = sy.Matrix([R_G2[1]]);

# 위치에너지 계산을 위해 y좌표만 뽑아낸다.
Y_H = R_H[1]
Y_G1 = R_G1[1]
Y_G2 = R_G2[1]

print(Y_H)
print(Y_G1)
print(Y_G2)

# 2) Lagrangian
T = 0.5*m*v_G1.dot(v_G1) + \
    0.5*m*v_G2.dot(v_G2) + \
    0.5*M*v_H.dot(v_H) + \
    0.5*I*omega1*omega1 + \
    0.5*I*(omega1+omega2)*(omega1+omega2)

V = m*g*Y_G1 + m*g*Y_G2 + M*g*Y_H
L = T-V
# L = sy.Matrix([L]);

print()
print(f'T: {T}')
print(f'V: {V}')
print(f'L: {L}')

# 3) Derive equations
# 오일러-라그랑지안 해석
qddot = sy.Matrix([ax, ay, alpha1, alpha2])
dLdqdot = []
ddt_dLdqdot = []
dLdq = []
EOM = []
mm = len(qddot)
for i in range(0, mm):
    dLdqdot.append(sy.diff(L, qdot[i]))
    tmp = 0
    for j in range(0, mm):
        tmp += sy.diff(dLdqdot[i], q[j]) * qdot[j] + sy.diff(dLdqdot[i], qdot[j]) * qddot[j]
    ddt_dLdqdot.append(tmp)
    dLdq.append(sy.diff(L, q[i]))
    EOM.append(ddt_dLdqdot[i] - dLdq[i])

EOM = sy.Matrix([EOM[0], EOM[1], EOM[2], EOM[3]])
print()
print(EOM)

# Equation for SS
# A_ss: alpha만 들어간 term / 4 * 4 크기를 갖는다.
# b_ss : 나머지 term
A_ss = EOM.jacobian(qddot)
b_ss = []
for i in range(0, mm):
    tmp = -EOM[i].subs([(ax, 0), (ay, 0), (alpha1, 0), (alpha2, 0)])
    b_ss.append(tmp)

# alpha 관련된 term들만 뽑아내고, 역행렬을 통해 각가속도만 계산한다.
print('A11 = ', sy.simplify(A_ss[2, 2]))
print('A12 = ', sy.simplify(A_ss[2, 3]))
print('A21 = ', sy.simplify(A_ss[3, 2]))
print('A22 = ', sy.simplify(A_ss[3, 3]), '\n')

print('b1 = ', sy.simplify(b_ss[2]))
print('b2 = ', sy.simplify(b_ss[3]), '\n')

print('A_ss = np.array([[A11, A12], [A21, A22]])')
print('b_ss = np.array([b1, b2])')
print('invA_ss = np.linalg.inv(A_ss)')
print('thetaddot = invA_ss.dot(b_ss)', '\n\n')

###############################
# Heelstrike stuff starts now #
###############################

# 점 C2에 대한 Jacobian 계산
r_C2 = sy.Matrix([C2[0], C2[1]])
J_sw = r_C2.jacobian(q)

# J_n_sw => 2 * 4 Matrix
# A_n_hs => 4 * 4 Matrix
J_n_sw = J_sw.subs([(theta1, theta1_n), (theta2, theta2_n)])
A_n_hs = A_ss.subs([(theta1, theta1_n), (theta2, theta2_n)])

# hs equations
print('J11 = ', sy.simplify(J_n_sw[0, 0]))
print('J12 = ', sy.simplify(J_n_sw[0, 1]))
print('J13 = ', sy.simplify(J_n_sw[0, 2]))
print('J14 = ', sy.simplify(J_n_sw[0, 3]))
print('J21 = ', sy.simplify(J_n_sw[1, 0]))
print('J22 = ', sy.simplify(J_n_sw[1, 1]))
print('J23 = ', sy.simplify(J_n_sw[1, 2]))
print('J24 = ', sy.simplify(J_n_sw[1, 3]), '\n')
print('J = np.array([[J11, J12, J13, J14], [J21,J22,J23,J24]])', '\n')

# A_n_hs는 단순히 theta1, theta2 대신에 theta1_n, theta2_n 넣은 것
print('A11 = ', sy.simplify(A_n_hs[0, 0]))
print('A12 = ', sy.simplify(A_n_hs[0, 1]))
print('A13 = ', sy.simplify(A_n_hs[0, 2]))
print('A14 = ', sy.simplify(A_n_hs[0, 3]))

print('A21 = ', sy.simplify(A_n_hs[1, 0]))
print('A22 = ', sy.simplify(A_n_hs[1, 1]))
print('A23 = ', sy.simplify(A_n_hs[1, 2]))
print('A24 = ', sy.simplify(A_n_hs[1, 3]))

print('A31 = ', sy.simplify(A_n_hs[2, 0]))
print('A32 = ', sy.simplify(A_n_hs[2, 1]))
print('A33 = ', sy.simplify(A_n_hs[2, 2]))
print('A34 = ', sy.simplify(A_n_hs[2, 3]))

print('A41 = ', sy.simplify(A_n_hs[3, 0]))
print('A42 = ', sy.simplify(A_n_hs[3, 1]))
print('A43 = ', sy.simplify(A_n_hs[3, 2]))
print('A44 = ', sy.simplify(A_n_hs[3, 3]))
print('A_n_hs = np.array([[A11, A12, A13, A14], [A21, A22, A23, A24], \
[A31, A32, A33, A34], [A41, A42, A43, A44]])\n')

# C2_dot의 state(-) => x_dot, y_dot, theta1_dot, theta2_dot
# X_n_hs = sy.Matrix([0, 0, omega1_n, omega2_n])
# # (4 * 4) * (4 * 1) => (4 * 1)
# b_temp = A_n_hs * X_n_hs
# b_hs = sy.Matrix([ b_temp[0], b_temp[1], b_temp[2], b_temp[3], 0, 0 ])
# zeros_22 = sy.zeros(2,2)

# A_hs = sy.zeros(6,6)
# A_hs[:4, :4] = A_n_hs
# A_hs[:4, 4:] = J_n_sw.T
# A_hs[4:, :4] = J_n_sw

# invA_hs = (A_hs**-1) * b_hs

print('X_n_hs = np.array([0, 0, omega1_n, omega2_n])')
print('b_temp  = A_n_hs.dot(X_n_hs)')
print('b_hs = np.block([ b_temp, 0, 0 ])')
print('zeros_22 = np.zeros((2,2))')
print('A_hs = np.block([[A_n_hs, -np.transpose(J)] , [ J, zeros_22] ])')
print('invA_hs = np.linalg.inv(A_hs)')

print('X_hs = invA_hs.dot(b_hs)')

# switch condition
print('omega1 = X_hs[2] + X_hs[3]')
print('omega2 = -X_hs[3]')
