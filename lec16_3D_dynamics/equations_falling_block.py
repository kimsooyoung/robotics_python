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


def sin(angle):
    return sy.sin(angle)


def cos(angle):
    return sy.cos(angle)


x, y, z = sy.symbols('x y z', real=True)
vx, vy, vz = sy.symbols('vx vy vz', real=True)

phi, theta, psi = sy.symbols('phi theta psi', real=True)
phi_d, theta_d, psi_d = sy.symbols('phi_d theta_d psi_d', real=True)

ax, ay, az = sy.symbols('ax ay az', real=True)
phi_dd, theta_dd, psi_dd = sy.symbols('phi_dd theta_dd psi_dd', real=True)

m, g, Ixx, Iyy, Izz = sy.symbols('m g Ixx Iyy Izz', real=True)

R_x = sy.Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])

R_y = sy.Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])

R_z = sy.Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])

i = sy.Matrix([1, 0, 0])
j = sy.Matrix([0, 1, 0])
k = sy.Matrix([0, 0, 1])

w_b = phi_d * i + theta_d * (R_x.T * j) + psi_d * (R_x.T * R_y.T * k)
w = psi_d * k + theta_d * (R_z * j) + phi_d * (R_z * R_y * i)

I = sy.Matrix([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
v = sy.Matrix([vx, vy, vz])

# Euler lagrange

# 주의 w_b.T * (I * w_b) 이렇게 쓰면 덧셈이 불가해짐
# 그래서 dot 연산을 사용하였다.
T = m / 2 * v.dot(v) + 1 / 2 * w_b.dot(I * w_b)
V = m * g * z
L = T - V

print('T: ', T)
print('V: ', V)

q = sy.Matrix([x, y, z, phi, theta, psi])
qdot = sy.Matrix([vx, vy, vz, phi_d, theta_d, psi_d])
qddot = sy.Matrix([ax, ay, az, phi_dd, theta_dd, psi_dd])
dLdqdot = []
ddt_dLdqdot = []
dLdq = []
EOM = []
mm = len(qddot)
for ii in range(0, mm):
    dLdqdot.append(sy.diff(L, qdot[ii]))
    tmp = 0
    for jj in range(0, mm):
        tmp += (
            sy.diff(dLdqdot[ii], q[jj]) * qdot[jj]
            + sy.diff(dLdqdot[ii], qdot[jj]) * qddot[jj]
        )
    ddt_dLdqdot.append(tmp)
    dLdq.append(sy.diff(L, q[ii]))
    EOM.append(ddt_dLdqdot[ii] - dLdq[ii])

EOM = sy.Matrix([EOM[0], EOM[1], EOM[2], EOM[3], EOM[4], EOM[5]])

A = EOM.jacobian(qddot)
b = []
for i in range(len(q)):
    temp = -EOM[i].subs(
        [(ax, 0), (ay, 0), (az, 0), (phi_dd, 0), (theta_dd, 0), (psi_dd, 0)]
    )
    b.append(temp)

m, n = A.shape
for i in range(m):
    for j in range(n):
        print(f'A[{i},{j}] = ', sy.simplify(A[i, j]))

m = len(b)
for i in range(m):
    print(f'b[{i}] = ', sy.simplify(b[i]))

angle_d = sy.Matrix([phi_d, theta_d, psi_d])
R_wb = w_b.jacobian(angle_d)
R_w = w.jacobian(angle_d)

m, n = R_w.shape
for i in range(m):
    for j in range(n):
        print(f'R_w[{i},{j}] = ', sy.simplify(R_w[i, j]))

m, n = R_wb.shape
for i in range(m):
    for j in range(n):
        print(f'R_wb[{i},{j}] = ', sy.simplify(R_wb[i, j]))
