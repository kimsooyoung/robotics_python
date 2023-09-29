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
theta1, theta2 = sy.symbols('theta1 theta2', real=True)
m1, m2 = sy.symbols('m1 m2', real=True)
l1, l2 = sy.symbols('l1 l2', real=True)
c1, c2 = sy.symbols('c1 c2', real=True)
I1, I2 = sy.symbols('I1 I2', real=True)
g = sy.symbols('g', real=True)

# position vectors
# link1 frame to world frame
theta1_ = 3*sy.pi/2 + theta1
H_01 = sy.Matrix([
    [sy.cos(theta1_), -sy.sin(theta1_), 0],
    [sy.sin(theta1_),  sy.cos(theta1_), 0],
    [0, 0, 1]
])

# link2 frame to world frame
H_12 = sy.Matrix([
    [sy.cos(theta2), -sy.sin(theta2), l1],
    [sy.sin(theta2),  sy.cos(theta2), 0],
    [0, 0, 1]
])
H_02 = H_01 * H_12

# c1 in world frame
G1 = H_01 * sy.Matrix([c1, 0, 1])
G1_xy = sy.Matrix([G1[0], G1[1]])

# c2 in world frame
G2 = H_02 * sy.Matrix([c2, 0, 1])
G2_xy = sy.Matrix([G2[0], G2[1]])

# velocity vectors
theta1_d, theta2_d = sy.symbols('theta1_d theta2_d', real=True)
q = sy.Matrix([theta1, theta2])
q_d = sy.Matrix([theta1_d, theta2_d])

v_G1 = G1_xy.jacobian(q) * q_d
v_G2 = G2_xy.jacobian(q) * q_d

# kinetic energy
T1 = 0.5*m1*v_G1.dot(v_G1) + 0.5*I1*theta1_d**2
T2 = 0.5*m2*v_G2.dot(v_G2) + 0.5*I2*(theta1_d+theta2_d)**2
T = T1 + T2

# potential energy
V1 = m1*g*G1[1]
V2 = m2*g*G2[1]
V = V1 + V2

# Lagrangian
L = T - V

# Lagrange equation
theta1_dd, theta2_dd = sy.symbols('theta1_dd theta2_dd', real=True)
q_dd = sy.Matrix([theta1_dd, theta2_dd])

dL_dq_d = []
dt_dL_dq_d = []
dL_dq = []
EOM = []

# Derive equations
for i in range(len(q)):
    dL_dq_d.append(sy.diff(L, q_d[i]))
    temp = 0
    for j in range(len(q)):
        temp += sy.diff(dL_dq_d[i], q[j]) * q_d[j] + \
                sy.diff(dL_dq_d[i], q_d[j]) * q_dd[j]

    dt_dL_dq_d.append(temp)
    dL_dq.append(sy.diff(L, q[i]))
    # 현재 외력이 0이므로 이 두개 항만 있다.
    EOM.append(dt_dL_dq_d[i] - dL_dq[i])

# EOM_0 = A11 theta1ddot + A12 theta2ddot - b1 = 0
# EOM_1 = A21 theta1ddot + A22 theta2ddot - b2 = 0
EOM = sy.Matrix([EOM[0], EOM[1]])
print('EOM_0 = ', sy.simplify(EOM[0]))
print('EOM_1 = ', sy.simplify(EOM[1]), '\n')

# M(q)*q_dd + C(q, q_d)*q_d + G(q) -Tau = 0
# C : coriolis force
# G : gravity
# M(q)*q_dd + C(q, q_d)*q_d + G(q) -Tau = 0
# b = C(q, q_d)*q_d + G(q) -Tau
# G = G(q)
# C = b - G = C(q, q_d)*q_d + G(q) - G(q) = C(q, q_d)*q_d
M = EOM.jacobian(q_dd)
b = EOM.subs([
    (theta1_dd, 0),
    (theta2_dd, 0),
])
G = b.subs([
    (theta1_d, 0),
    (theta2_d, 0),
])
C = b - G

print('M11 = ', sy.simplify(M[0, 0]))
print('M12 = ', sy.simplify(M[0, 1]))
print('M21 = ', sy.simplify(M[1, 0]))
print('M22 = ', sy.simplify(M[1, 1]), '\n')

print('C1 = ', sy.simplify(C[0]))
print('C2 = ', sy.simplify(C[1]), '\n')
print('G1 = ', sy.simplify(G[0]))
print('G2 = ', sy.simplify(G[1]), '\n')
