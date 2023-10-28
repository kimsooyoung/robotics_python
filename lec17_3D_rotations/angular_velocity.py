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


phi, theta, psi = sy.symbols('phi theta psi', real=True)
phidot, thetadot, psidot = sy.symbols('phidot thetadot psidot', real=True)

# unit vectors
i = sy.Matrix([1, 0, 0])
j = sy.Matrix([0, 1, 0])
k = sy.Matrix([0, 0, 1])

R_x = sy.Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])

R_y = sy.Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])

R_z = sy.Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
q_d = sy.Matrix([phidot, thetadot, psidot])


# Get angular velocity in the world frame
om = psidot * k + R_z * (thetadot * j) + R_z * R_y * (phidot * i)
# w = R_we * q_dot임을 시용해서 R_we만 추출
R_we = om.jacobian(q_d)
R_we = sy.simplify(R_we)
print(sy.simplify(R_we))
print(sy.simplify(R_we.det()), '\n')

# Get angular velocity in the body frame
om_b = (
    phidot * i
    + R_x.transpose() * (thetadot * j)
    + R_x.transpose() * R_y.transpose() * (psidot * k)
)
R_be = om_b.jacobian(q_d)
print(sy.simplify(R_be))
print(sy.simplify(R_be.det()), '\n')

# Get linear velocity in the world frame
rx, ry, rz = sy.symbols('rx ry rz', real=True)
r = rx * i + ry * j + rz * k
V = om.cross(r)
print(sy.simplify(V), '\n')

# Get linear velocity in the body frame
V_b = om_b.cross(r)
print(sy.simplify(V_b), '\n')
