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

# define variables
t = sy.symbols('t', real=True)
a10, a11, a12, a13 = sy.symbols('a10 a11 a12 a13', real=True)
a20, a21, a22, a23 = sy.symbols('a20 a21 a22 a23', real=True)

pi = sy.pi

f0 = a10 + a11*t + a12*t**2 + a13*t**3
f1 = a20 + a21*t + a22*t**2 + a23*t**3

t0, t1, t2 = 0.0, 1.5, 3.0
theta0, theta1, theta2 = 0.0, 0.5*pi, 0.0

f0_d, f1_d = sy.diff(f0, t), sy.diff(f1, t)
f0_dd, f1_dd = sy.diff(f0_d, t), sy.diff(f1_d, t)

# theta1(0) = 0 / theta1(1.5) = 0.5*pi
# theta2(1.5) = 0.5*pi / theta(3) = 0
# theta1_d(0) = 0 / theta2_d(3) = 0
# theta1_d(1.5) = theta2_d(1.5) / theta1_dd(1.5) = theta2_dd(1.5)
equ0 = f0.subs(t, t0) - theta0
equ1 = f0.subs(t, t1) - theta1
equ2 = f1.subs(t, t1) - theta1
equ3 = f1.subs(t, t2) - theta2
equ4 = f0_d.subs(t, t0) - 0
equ5 = f1_d.subs(t, t2) - 0
equ6 = f1_d.subs(t, t1) - f0_d.subs(t, t1)
equ7 = f1_dd.subs(t, t1) - f0_dd.subs(t, t1)

# Ax = b 만들어보자.
q = sy.Matrix([a10, a11, a12, a13, a20, a21, a22, a23])
equ = sy.Matrix([equ0, equ1, equ2, equ3, equ4, equ5, equ6, equ7])

A = equ.jacobian(q)
b = -equ.subs([
    (a10, 0), (a11, 0), (a12, 0), (a13, 0),
    (a20, 0), (a21, 0), (a22, 0), (a23, 0)
])

x = A.inv() * b

# print all elements of x in a nice way
print(f'a10 = {x[0]}')
print(f'a11 = {x[1]}')
print(f'a12 = {x[2]}')
print(f'a13 = {x[3]}')
print(f'a20 = {x[4]}')
print(f'a21 = {x[5]}')
print(f'a22 = {x[6]}')
print(f'a23 = {x[7]}')
