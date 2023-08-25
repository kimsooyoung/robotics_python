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

# defining the variables
pi = sy.pi
t = sy.symbols('t', real=True)
a10, a11, a12, a13 = sy.symbols('a10 a11 a12 a13', real=True)
a20, a21, a22, a23 = sy.symbols('a20 a21 a22 a23', real=True)

pose_1 = a10 + a11*t + a12*t**2 + a13*t**3
pose_2 = a20 + a21*t + a22*t**2 + a23*t**3

vel_1 = a11 + 2*a12*t + 3*a13*t**2
vel_2 = a21 + 2*a22*t + 3*a23*t**2

acc_1 = 2*a12 + 6*a13*t
acc_2 = 2*a22 + 6*a23*t

eqn1 = pose_1.subs(t, 0) - 0
eqn2 = pose_1.subs(t, 1.5) - (-pi/2 + 0.5)
eqn3 = pose_2.subs(t, 1.5) - (-pi/2 + 0.5)
eqn4 = pose_2.subs(t, 3) - 0
eqn5 = vel_1.subs(t, 0) - 0
eqn6 = vel_2.subs(t, 3) - 0
eqn7 = vel_1.subs(t, 1.5) - vel_2.subs(t, 1.5)
eqn8 = acc_1.subs(t, 1.5) - acc_2.subs(t, 1.5)

eqn = sy.Matrix([eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8])
q = sy.Matrix([a10, a11, a12, a13, a20, a21, a22, a23])

A = eqn.jacobian(q)
b = -eqn.subs([(a10, 0), (a11, 0), (a12, 0), (a13, 0), (a20, 0), (a21, 0), (a22, 0), (a23, 0)])

x = A.inv()*b

print('a10 = ', x[0])
print('a11 = ', x[1])
print('a12 = ', x[2])
print('a13 = ', x[3])
print('a20 = ', x[4])
print('a21 = ', x[5])
print('a22 = ', x[6])
print('a23 = ', x[7])
