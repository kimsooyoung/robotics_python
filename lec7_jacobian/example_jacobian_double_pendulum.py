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
c1, c2, l = sy.symbols('c1 c2 l', real=True)

H_01 = sy.Matrix([
    [sy.cos(3*sy.pi/2 + theta1), -sy.sin(3*sy.pi/2 + theta1), 0],
    [sy.sin(3*sy.pi/2 + theta1),  sy.cos(3*sy.pi/2 + theta1), 0],
    [0, 0, 1]
])

H_12 = sy.Matrix([
    [sy.cos(theta2), -sy.sin(theta2), l],
    [sy.sin(theta2),  sy.cos(theta2), 0],
    [0, 0, 1]
])

H_02 = H_01 * H_12

G1_1 = sy.Matrix([c1, 0, 1])
G1_0 = H_01 * G1_1
G1_0.row_del(2)

G2_2 = sy.Matrix([c2, 0, 1])
G2_0 = H_02 * G2_2
G2_0.row_del(2)

q = sy.Matrix([theta1, theta2])
# Jacobian of link1 COM
J_G1 = G1_0.jacobian(q)
# Jacobian of link2 COM
J_G2 = sy.simplify(G2_0.jacobian(q))

print(J_G1)
print(J_G2)

# Application1 - cartesian velocity
omega1, omega2 = sy.symbols('omega1 omega2', real=True)
q_dot = sy.Matrix([omega1, omega2])

V_G1 = sy.simplify(J_G1 * q_dot)
V_G2 = sy.simplify(J_G2 * q_dot)
print('V_G1, V_G2')
print(V_G1)
print(V_G2)

# Application2 - static forces
m1, m2, g = sy.symbols('m1 m2 g', real=True)
F1 = sy.Matrix([0, -m1*g])
F2 = sy.Matrix([0, -m2*g])
tau = sy.simplify(J_G1.transpose()*F1) + sy.simplify(J_G2.transpose()*F2)

print(f'tau0 = {tau[0]}')
print(f'tau1 = {tau[1]}')


# Application3 - inverse kinematics

# End point
E2_2 = sy.Matrix([l, 0, 1])
E2_0 = H_02 * E2_2
E2_0.row_del(2)

# Jacobian of End Point
E_G2 = sy.simplify(E2_0.jacobian(q))
print(f'E_Jacobi00 = {E_G2[0,0]}')
print(f'E_Jacobi01 = {E_G2[0,1]}')
print(f'E_Jacobi10 = {E_G2[1,0]}')
print(f'E_Jacobi11 = {E_G2[1,1]}')
