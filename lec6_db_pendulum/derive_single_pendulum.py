# Copyright 2024 @RoadBalance
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
theta = sy.symbols('theta', real=True)
m, l, c, I = sy.symbols('m l c I', real=True)
g = sy.symbols('g', real=True)

# position vectors
# link1 frame to world frame
theta_ = 3*sy.pi/2 + theta
H_01 = sy.Matrix([
    [sy.cos(theta_), -sy.sin(theta_), 0],
    [sy.sin(theta_),  sy.cos(theta_), 0],
    [0, 0, 1]
])

# Pendulum COM in world frame
G = H_01 * sy.Matrix([c, 0, 1])
G_xy = sy.Matrix([G[0], G[1]])

# velocity vectors
theta_d = sy.symbols('theta_d', real=True)
q = sy.Matrix([theta])
q_d = sy.Matrix([theta_d])

v_G = G_xy.jacobian(q) * q_d

# kinetic energy
T = 0.5*m*v_G.dot(v_G) + 0.5*I*theta_d**2
# T = 0.5*I*theta_d**2

# potential energy
V = m*g*G[1]

# Lagrangian
L = T - V

# Lagrange equation
theta_dd = sy.symbols('theta_dd', real=True)
q_dd = sy.Matrix([theta_dd])

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

EOM = sy.Matrix([EOM])
print(f"EOM: {EOM}")

# M(q)*q_dd + C(q, q_d)*q_d + G(q) -Tau = 0
# C : coriolis force
# G : gravity
# M(q)*q_dd + C(q, q_d)*q_d + G(q) -Tau = 0
# b = C(q, q_d)*q_d + G(q) -Tau
# G = G(q)
# C = b - G = C(q, q_d)*q_d + G(q) - G(q) = C(q, q_d)*q_d
M = EOM.jacobian(q_dd)
b = EOM.subs([
    (theta_dd, 0)
])
G = b.subs([
    (theta_d, 0),
])
C = b - G

print(f"M : {M[0]}")
print(f"C : {C[0]}")
print(f"G : {G[0]}")
