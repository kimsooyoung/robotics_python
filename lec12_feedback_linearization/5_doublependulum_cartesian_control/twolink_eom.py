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
theta1dot, theta2dot = sy.symbols('theta1dot theta2dot', real=True)
theta1ddot, theta2ddot = sy.symbols('theta1ddot theta2ddot', real=True)
m1, m2, I1, I2, g = sy.symbols('m1 m2 I1 I2 g', real=True)
c1, c2, l1, l2 = sy.symbols('c1 c2 l1 l2', real=True)

# 1a) position vectors
mpi = sy.pi
cos1 = sy.cos(theta1)
sin1 = sy.sin(theta1)
H01 = sy.Matrix([[cos1, -sin1, 0], [sin1, cos1, 0], [0, 0, 1]])

cos2 = sy.cos(theta2)
sin2 = sy.sin(theta2)
H12 = sy.Matrix([[cos2, -sin2, l1], [sin2, cos2, 0], [0, 0, 1]])
H02 = H01 * H12

C1 = sy.Matrix([c1, 0, 1])
G1 = H01 * C1
C2 = sy.Matrix([c2, 0, 1])
G2 = H02 * C2
E = sy.Matrix([l2, 0, 1])
E = H02 * E

x_G1 = sy.Matrix([G1[0]])
y_G1 = sy.Matrix([G1[1]])
x_G2 = sy.Matrix([G2[0]])
y_G2 = sy.Matrix([G2[1]])

# 1b) velocity vectors
q = sy.Matrix([theta1, theta2])
qdot = sy.Matrix([theta1dot, theta2dot])
v_G1_x = x_G1.jacobian(q) * qdot
v_G1_y = y_G1.jacobian(q) * qdot
v_G2_x = x_G2.jacobian(q) * qdot
v_G2_y = y_G2.jacobian(q) * qdot
v_G1 = sy.Matrix([v_G1_x, v_G1_y])
v_G2 = sy.Matrix([v_G2_x, v_G2_y])

# 2) Lagrangian
T = (
    0.5 * m1 * v_G1.dot(v_G1)
    + 0.5 * m2 * v_G2.dot(v_G2)
    + 0.5 * I1 * theta1dot * theta1dot
    + 0.5 * I2 * (theta1dot + theta2dot) * (theta1dot + theta2dot)
)
V = m1 * g * G1[1] + m2 * g * G2[1]
L = T - V
# print(L)
# print(type(T))
# print(type(V))
# print(type(L))

# 3) Derive equations
qddot = sy.Matrix([theta1ddot, theta2ddot])
dLdqdot = []
ddt_dLdqdot = []
dLdq = []
EOM = []
mm = len(qddot)
for i in range(0, mm):
    dLdqdot.append(sy.diff(L, qdot[i]))
    tmp = 0
    for j in range(0, mm):
        tmp += (
            sy.diff(dLdqdot[i], q[j]) * qdot[j]
            + sy.diff(dLdqdot[i], qdot[j]) * qddot[j]
        )
    ddt_dLdqdot.append(tmp)
    dLdq.append(sy.diff(L, q[i]))
    EOM.append(ddt_dLdqdot[i] - dLdq[i])

EOM = sy.Matrix([EOM[0], EOM[1]])
# print(len(EOM))
# print(type(qddot))
# print(type(EOM))
M = EOM.jacobian(qddot)
N1 = EOM[0].subs([(theta1ddot, 0), (theta2ddot, 0)])
N2 = EOM[1].subs([(theta1ddot, 0), (theta2ddot, 0)])
G1 = N1.subs([(theta1dot, 0), (theta2dot, 0)])
G2 = N2.subs([(theta1dot, 0), (theta2dot, 0)])
C1 = N1 - G1
C2 = N2 - G2

# print(EOM.shape)
# print(M.shape)
print('M11 = ', sy.simplify(M[0, 0]))
print('M12 = ', sy.simplify(M[0, 1]))
print('M21 = ', sy.simplify(M[1, 0]))
print('M22 = ', sy.simplify(M[1, 1]), '\n')

print('C1 = ', sy.simplify(C1))
print('C2 = ', sy.simplify(C2), '\n')
print('G1 = ', sy.simplify(G1))
print('G2 = ', sy.simplify(G2), '\n')

Tip = sy.Matrix([E[0], E[1]])
J = Tip.jacobian(q)
print('J11 = ', sy.simplify(J[0, 0]))
print('J12 = ', sy.simplify(J[0, 1]))
print('J21 = ', sy.simplify(J[1, 0]))
print('J22 = ', sy.simplify(J[1, 1]), '\n')

Jdot = sy.Matrix([[0, 0], [0, 0]])
Jdot[0, 0] = sy.diff(J[0, 0], theta1) * theta1dot + sy.diff(J[0, 0], theta2) * theta2dot
Jdot[0, 1] = sy.diff(J[0, 1], theta1) * theta1dot + sy.diff(J[0, 1], theta2) * theta2dot
Jdot[1, 0] = sy.diff(J[1, 0], theta1) * theta1dot + sy.diff(J[1, 0], theta2) * theta2dot
Jdot[1, 1] = sy.diff(J[1, 1], theta1) * theta1dot + sy.diff(J[1, 1], theta2) * theta2dot
print('Jdot11 = ', sy.simplify(Jdot[0, 0]))
print('Jdot12 = ', sy.simplify(Jdot[0, 1]))
print('Jdot21 = ', sy.simplify(Jdot[1, 0]))
print('Jdot22 = ', sy.simplify(Jdot[1, 1]), '\n')
