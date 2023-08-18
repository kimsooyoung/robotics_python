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
import numpy as np 

# define state variables
# 각도, 각속도, 각가속도
theta1, theta2 = sy.symbols('theta1 theta2', real=True)
omega1, omega2 = sy.symbols('omega1 omega2', real=True)
alpha1, alpha2 = sy.symbols('alpha1 alpha2', real=True)

# 충돌 후의 각도와 각속도
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

##############################
##### Step 1. kinematics #####
##############################

pi = sy.pi
cos = sy.cos
sin = sy.sin

angle_1 = pi/2 + theta1

# floating base => x, y, 1
H_01 = sy.Matrix([
    [cos(angle_1), -sin(angle_1), x],
    [sin(angle_1), cos(angle_1), y],
    [0, 0, 1]
])

# 각도 유의
angle_2 = theta2 - pi
H_12 = sy.Matrix([
    [cos(angle_2), -sin(angle_2), l],
    [sin(angle_2), cos(angle_2), 0],
    [0, 0, 1]
])

H_02 = H_01 * H_12

C1 = sy.Matrix([x, y, 1])
H  = H_01 * sy.Matrix([l, 0, 1])
C2 = H_02 * sy.Matrix([l, 0, 1])

G1 = H_01 * sy.Matrix([l - c, 0, 1])
G2 = H_02 * sy.Matrix([c, 0, 1])

# print(f"H: {H}")
# print(f"G1: {G1}")
# print(f"G2: {G2}")
# print(f"C1: {C1}")
# print(f"C2: {C2}")

##############################
#####  Step 2. velocity  #####
##############################

q = sy.Matrix([x, y, theta1, theta2])
q_d = sy.Matrix([vx, vy, omega1, omega2])

# 의문 => 왜 여기서는 [x, y, theta1, theta2]로 잡은거지?
# double pendulum에서는 [theta1, theta2]였고 이것만으로도 운동에너지 구했다.
# 
# => double pendulum에서는 고정된 점이 있었는데, 지금은 모두 움직여서 그런가?
# A) impact map을 고려해야 하니 x, y term도 추가한 것임

H_xy = sy.Matrix([H[0], H[1]])
G1_xy = sy.Matrix([G1[0], G1[1]])
G2_xy = sy.Matrix([G2[0], G2[1]])

v_H = H_xy.jacobian(q) * q_d
v_G1 = G1_xy.jacobian(q) * q_d
v_G2 = G2_xy.jacobian(q) * q_d

# print()
# print(f"v_H: {v_H}")
# print(f"v_G1: {v_G1}")
# print(f"v_G2: {v_G2}")

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
R_G1 = H_og * G1
R_G2 = H_og * G2

# print(R_H[1])
# print(R_G1[1])
# print(R_G2[1])

T = 0.5 * M * v_H.dot(v_H) + \
    0.5 * m * v_G1.dot(v_G1) + \
    0.5 * m * v_G2.dot(v_G2) + \
    0.5 * I * omega1**2 + \
    0.5 * I * (omega1 + omega2) **2

V = m * g * R_G1[1] + \
    m * g * R_G2[1] + \
    M * g * R_H[1]

L = T - V

# print(f"T: {T}")
# print(f"V: {V}")
# print(f"L: {L}")

# Lagrange Equation
dL_dq_d = []
dt_dL_dq_d = []
dL_dq = []
q_dd = sy.Matrix([ax, ay, alpha1, alpha2])

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
# print(EOM)

#################################
##### Step 4. single_stance #####
#################################

# alpha1, alpha2를 얻어내는 것이 목표
# 사실 이것을 위해서는 x,y관련된 term은 필요가 없다.
# 따라서 EOM matrix의 오른쪽 하단 부분만 추출해서 사용한다.

# Ax = b
A_ss = EOM.jacobian(q_dd)
b_ss = []

for i in range(len(q_dd)):
    b_ss.append(-1 * EOM[i].subs([(ax, 0), (ay, 0), (alpha1, 0), (alpha2, 0)]))

print(f"A_ss[2,2]: {sy.simplify(A_ss[2,2])}")
print(f"A_ss[2,3]: {sy.simplify(A_ss[2,3])}")
print(f"A_ss[3,2]: {sy.simplify(A_ss[3,2])}")
print(f"A_ss[3,3]: {sy.simplify(A_ss[3,3])}")

# # when real problem, use this
# print("A_ss = np.array([ [A22, A23], [A32, A33] ])")
# print("b_ss = np.array([ b2, b3 ])")
# print("q_dd = np.linalg.inv(A_ss).dot(b_ss)")

##############################
##### Step 5. Heelstrike #####
##############################

# 강의노트를 참고하면, 구속 조건이 있을 때의 EOM
# 즉 DAE를 세우는 방법을 알 수 있다.
# 해당 식을 위해서는 C2의 Jacobian, Matrix M이 필요하다.
# 목표는 strike 이후의 각도와 각속도를 얻어내는 것이다.

C2_xy = sy.Matrix([ C2[0], C2[1] ])
J_C2 = C2_xy.jacobian(q)

A_n_hs = A_ss.subs([ (theta1, theta1_n), (theta2, theta2_n) ])
J_n_sw = J_C2.subs([ (theta1, theta1_n), (theta2, theta2_n) ])

#hs equations
print('J11 = ', sy.simplify(J_n_sw[0,0]))
print('J12 = ', sy.simplify(J_n_sw[0,1]))
print('J13 = ', sy.simplify(J_n_sw[0,2]))
print('J14 = ', sy.simplify(J_n_sw[0,3]))
print('J21 = ', sy.simplify(J_n_sw[1,0]))
print('J22 = ', sy.simplify(J_n_sw[1,1]))
print('J23 = ', sy.simplify(J_n_sw[1,2]))
print('J24 = ', sy.simplify(J_n_sw[1,3]),'\n')

# print('J = np.array([[J11, J12, J13, J14], [J21,J22,J23,J24]])','\n');

# A_n_hs는 단순히 theta1, theta2 대신에 theta1_n, theta2_n 넣은 것
print('A11 = ', sy.simplify(A_n_hs[0,0]))
print('A12 = ', sy.simplify(A_n_hs[0,1]))
print('A13 = ', sy.simplify(A_n_hs[0,2]))
print('A14 = ', sy.simplify(A_n_hs[0,3]))

print('A21 = ', sy.simplify(A_n_hs[1,0]))
print('A22 = ', sy.simplify(A_n_hs[1,1]))
print('A23 = ', sy.simplify(A_n_hs[1,2]))
print('A24 = ', sy.simplify(A_n_hs[1,3]))

print('A31 = ', sy.simplify(A_n_hs[2,0]))
print('A32 = ', sy.simplify(A_n_hs[2,1]))
print('A33 = ', sy.simplify(A_n_hs[2,2]))
print('A34 = ', sy.simplify(A_n_hs[2,3]))

print('A41 = ', sy.simplify(A_n_hs[3,0]))
print('A42 = ', sy.simplify(A_n_hs[3,1]))
print('A43 = ', sy.simplify(A_n_hs[3,2]))
print('A44 = ', sy.simplify(A_n_hs[3,3]))

# Ax = b를 다시 세워보자. (여기서부터는 numpy가 사용된다.)
#
# z_d(+)와 I_c2가 목표이다.
# [ [ M -J_c.T ], [ J_c.T 0 ] ] * [ z_d(+) I_c2 ] = [ M*z_d(-) 0 ]

# 충돌 후 vx, vy는 0이며, 각속도는 구해야 한다.
# M*z_d(-) => A_n_hs.dot([0, 0, omega1_n, omega2_n])

"""
A_hs = np.block([
    [A_n_hs, -np.transpose(J_n_sw) ], 
    [J_n_sw, np.zeros((2,2))] 
])

b_hs = np.block([
    A_n_hs.dot([0, 0, omega1_n, omega2_n]),
    np.zeros((2,1))
])

# x_hs => [vx(+), vy(+), omega1(+), omega2(+) ]
x_hs = np.linalg.inv(A_hs).dot(b_hs)

# switch condition
omega1 = x_hs[2] + x_hs[3]
omega2 = -x_hs[3]
"""