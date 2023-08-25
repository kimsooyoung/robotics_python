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

m, g, l, I = sy.symbols('m g l I')
u1, u2 = sy.symbols('u1 u2')
x, y, phi = sy.symbols('x y phi')
x_d, y_d, phi_d = sy.symbols('x_d y_d phi_d')
x_dd, y_dd, phi_dd = sy.symbols('x_dd y_dd phi_dd')


T = (x_d**2 + y_d**2) * (m / 2) + (phi_d**2) * (I / 2)
V = m * g * y
L = T - V

H_g0 = sy.Matrix([[1, 0, x], [0, 1, y], [0, 0, 1]])

H_01 = sy.Matrix(
    [[sy.cos(phi), -sy.sin(phi), 0], [sy.sin(phi), sy.cos(phi), 0], [0, 0, 1]]
)

H_g1 = H_g0 * H_01

P = sy.Matrix([l / 2, 0, 1])
R = sy.Matrix([-l / 2, 0, 1])

P_0 = H_g1 * P
R_0 = H_g1 * R

P_0 = sy.Matrix([P_0[0], P_0[1]])
R_0 = sy.Matrix([R_0[0], R_0[1]])

R_op = sy.Matrix([[sy.cos(phi), -sy.sin(phi)], [sy.sin(phi), sy.cos(phi)]])

F_op = R_op * sy.Matrix([0, u1])
F_or = R_op * sy.Matrix([0, u2])

q = sy.Matrix([x, y, phi])

J_p = P_0.jacobian(q)
J_r = R_0.jacobian(q)

Q = sy.simplify(J_p.T * F_op + J_r.T * F_or)

# EOM
q_d = sy.Matrix([x_d, y_d, phi_d])
q_dd = sy.Matrix([x_dd, y_dd, phi_dd])

EOM = []
EOM_control = []
dLdq = []
dLdqd = []
dt_dLdqd = []

for i in range(len(q)):
    dLdq.append(sy.diff(L, q[i]))
    dLdqd.append(sy.diff(L, q_d[i]))

    temp = 0
    for j in range(len(q)):
        temp += sy.diff(dLdqd[i], q[j]) * q_d[j] + sy.diff(dLdqd[i], q_d[j]) * q_dd[j]
    dt_dLdqd.append(temp)

    EOM.append(dt_dLdqd[i] - dLdq[i])
    EOM_control.append(dt_dLdqd[i] - dLdq[i] - Q[i])

print('x_dd = ', (sy.solve(EOM_control[0], x_dd)[0]))
print('y_dd = ', (sy.solve(EOM_control[1], y_dd)[0]))
print('phi_dd = ', (sy.solve(EOM_control[2], phi_dd)[0]))

# TODO: LQR controller
EOM = sy.Matrix([EOM[0], EOM[1], EOM[2]])
M = EOM.jacobian(q_dd)
N1 = EOM[0].subs([(x_dd, 0), (y_dd, 0), (phi_dd, 0)])
N2 = EOM[1].subs([(x_dd, 0), (y_dd, 0), (phi_dd, 0)])
N3 = EOM[2].subs([(x_dd, 0), (y_dd, 0), (phi_dd, 0)])
G1 = N1.subs([(x_d, 0), (y_d, 0), (phi_d, 0)])
G2 = N2.subs([(x_d, 0), (y_d, 0), (phi_d, 0)])
G3 = N3.subs([(x_d, 0), (y_d, 0), (phi_d, 0)])
C1 = N1 - G1
C2 = N2 - G2
C3 = N3 - G3

print('M11 = ', sy.simplify(M[0, 0]))
print('M12 = ', sy.simplify(M[0, 1]))
print('M13 = ', sy.simplify(M[0, 2]))
print('M21 = ', sy.simplify(M[1, 0]))
print('M22 = ', sy.simplify(M[1, 1]))
print('M23 = ', sy.simplify(M[1, 2]))
print('M31 = ', sy.simplify(M[2, 0]))
print('M32 = ', sy.simplify(M[2, 1]))
print('M33 = ', sy.simplify(M[2, 2]), '\n')

print('C1 = ', sy.simplify(C1))
print('C2 = ', sy.simplify(C2))
print('C3 = ', sy.simplify(C3), '\n')

print('G1 = ', sy.simplify(G1))
print('G2 = ', sy.simplify(G2))
print('G3 = ', sy.simplify(G3), '\n')
