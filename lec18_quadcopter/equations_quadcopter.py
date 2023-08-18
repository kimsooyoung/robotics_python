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


def cos(angle):
    return sy.cos(angle)


def sin(angle):
    return sy.sin(angle)


x, y, z = sy.symbols('x y z', real=True)
vx, vy, vz = sy.symbols('vx vy vz', real=True)
ax, ay, az = sy.symbols('ax ay az', real=True)

phi, theta, psi = sy.symbols('phi theta psi', real=True)
phi_d, theta_d, psi_d = sy.symbols('phi_d theta_d psi_d', real=True)
phi_dd, theta_dd, psi_dd = sy.symbols('phi_dd theta_dd psi_dd', real=True)

m, g, l = sy.symbols('m g l', real=True)
Ixx, Iyy, Izz = sy.symbols('Ixx Iyy Izz', real=True)

K, b = sy.symbols('K b', real=True)
Ax, Ay, Az = sy.symbols('Ax Ay Az', real=True)
omega1, omega2, omega3, omega4 = sy.symbols('omega1 omega2 omega3 omega4', real=True)

# 1) position and angles
R_x = sy.Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])

R_y = sy.Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])

R_z = sy.Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])

# Rotation body frame to world frame
R_b2w = R_z * R_y * R_x

i = sy.Matrix([1, 0, 0])
j = sy.Matrix([0, 1, 0])
k = sy.Matrix([0, 0, 1])

I = sy.Matrix([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
v = sy.Matrix([vx, vy, vz])

# 2) angular velocity and energy
# omega in body frame
w_b = phi_d * i + theta_d * (R_x.T * j) + psi_d * (R_x.T * R_y.T * k)
# omega in world frame
w = psi_d * k + theta_d * (R_z * j) + phi_d * (R_z * R_y * i)

# EL method
# 주의 w_b.T * (I * w_b) 이렇게 쓰면 덧셈이 불가해짐
T = m / 2 * (v.dot(v)) + 1 / 2 * (w_b.dot(I * w_b))
V = m * g * z
L = T - V

# print(T)
# print(V)

# external force
Trust = sy.Matrix([0, 0, K * (omega1**2 + omega2**2 + omega3**2 + omega4**2)])
DragForce = sy.Matrix([Ax * vx, Ay * vy, Az * vz])
NetForce = R_b2w * Trust - DragForce

# external torque
TrustTorque = sy.Matrix(
    [
        K * l / 2 * (omega4**2 - omega2**2),
        K * l / 2 * (omega3**2 - omega1**2),
        b * (omega1**2 - omega2**2 + omega3**2 - omega4**2),
    ]
)

Q_j = NetForce.col_join(TrustTorque)
# print(Q_j)

# 3) Derive equations
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
    EOM.append(ddt_dLdqdot[ii] - dLdq[ii] - Q_j[ii])

EOM = sy.Matrix([EOM[0], EOM[1], EOM[2], EOM[3], EOM[4], EOM[5]])

A = EOM.jacobian(qddot)
m, n = A.shape

for i in range(m):
    for j in range(n):
        print('A[', i, ',', j, ']=', sy.simplify(A[i, j]))

B = []
for i in range(m):
    temp = (
        -EOM[i]
        .subs(qddot[0], 0)
        .subs(qddot[1], 0)
        .subs(qddot[2], 0)
        .subs(qddot[3], 0)
        .subs(qddot[4], 0)
        .subs(qddot[5], 0)
    )
    B.append(temp)

for i in range(len(B)):
    print('B[', i, ']=', sy.simplify(B[i]))

# world frame velocity
angle_d = sy.Matrix([phi_d, theta_d, psi_d])
R_w = w.jacobian(angle_d)
R_wb = w_b.jacobian(angle_d)

m, n = R_w.shape

for i in range(m):
    for j in range(n):
        print('R_w[', i, ',', j, ']=', sy.simplify(R_w[i, j]))

for i in range(m):
    for j in range(n):
        print('R_wb[', i, ',', j, ']=', sy.simplify(R_wb[i, j]))
