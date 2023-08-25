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

theta1, theta2 = sy.symbols('theta1 theta2', real=True)
pi = sy.pi
c1, c2, l, m1, m2 = sy.symbols('c1 c2 l m1 m2', real=True)

H_01 = sy.Matrix([
    [sy.cos(theta1), -sy.sin(theta1), 0],
    [sy.sin(theta1),  sy.cos(theta1), 0],
    [0, 0, 1]
])

H_12 = sy.Matrix([
    [sy.cos(theta2), -sy.sin(theta2), l],
    [sy.sin(theta2),  sy.cos(theta2), 0],
    [0, 0, 1]
])

H_02 = H_01 * H_12

C1 = sy.Matrix([c1, 0, 1])
C2 = sy.Matrix([c2, 0, 1])

# global frame에서의 좌표들
G1 = H_01 * C1
G2 = H_02 * C2

G1_xy = sy.Matrix([G1[0], G1[1]])
G2_xy = sy.Matrix([G2[0], G2[1]])
# print(G1_xy)
# print(G2_xy)

omega1, omega2 = sy.symbols('omega1 omega2', real=True)
I1, I2, g = sy.symbols('I1 I2 g', real=True)

q = sy.Matrix([theta1, theta2])
q_d = sy.Matrix([omega1, omega2])

V1 = G1_xy.jacobian(q) * q_d
V2 = G2_xy.jacobian(q) * q_d
# print(sy.simplify(V1))
# print(sy.simplify(V2))

T = m1/2*V1.dot(V1) + m2/2*V2.dot(V2) + I1/2*omega1**2 + I2/2*(omega1+omega2)**2
V = m1*g*G1_xy[1] + m2*g*G2_xy[1]
L = T - V
# print(sy.simplify(L))

a_acc1, a_acc2 = sy.symbols('a_acc1 a_acc2', real=True)
q_dd = sy.Matrix([a_acc1, a_acc2])
dL_dq = []
dL_dq_d = []
dL_dq_d_dt = []
EOM = []

for i in range(len(q)):
    dL_dq.append(sy.diff(L, q[i]))
    dL_dq_d.append(sy.diff(L, q_d[i]))

    temp = 0
    for j in range(len(q_d)):
        temp += sy.diff(dL_dq_d[i], q[j])*q_d[j] + \
                sy.diff(dL_dq_d[i], q_d[j])*q_dd[j]

    dL_dq_d_dt.append(temp)
    EOM.append(dL_dq_d_dt[i] - dL_dq[i])

EOM = sy.Matrix([EOM[0], EOM[1]])
# print(sy.simplify(EOM))

# M + C + G = tau

M = EOM.jacobian(q_dd)
C_G = EOM.subs([(a_acc1, 0), (a_acc2, 0)])
G = C_G.subs([(omega1, 0), (omega2, 0)])
C = C_G - G

print('M11 = ', sy.simplify(M[0, 0]))
print('M12 = ', sy.simplify(M[0, 1]))
print('M21 = ', sy.simplify(M[1, 0]))
print('M22 = ', sy.simplify(M[1, 1]), '\n')

print('C1 = ', sy.simplify(C[0]))
print('C2 = ', sy.simplify(C[1]), '\n')
print('G1 = ', sy.simplify(G[0]))
print('G2 = ', sy.simplify(G[1]), '\n')
