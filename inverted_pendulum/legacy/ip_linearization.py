# Copyright 2023 @RoadBalance
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

# define state variables
u = sy.symbols('u', real=True)
x, x_dot, x_ddot = sy.symbols('x x_dot x_ddot', real=True)
theta, theta_dot, theta_ddot = sy.symbols('theta theta_dot theta_ddot', real=True)

# m, M : pendulum mass, cart mass
# L : pendulum length
# g : gravity
# d : damping
m, M, L = sy.symbols('m M L', real=True)
g, d = sy.symbols('g d', real=True)

D = m*L*L*(M+m*(1-sy.cos(theta)**2))
Sx = sy.sin(theta)
Cx = sy.cos(theta)

#############################
# EOM from Control bootcamp #
#############################

A1 = x_dot
A2 = (1/D)*(-m**2*L**2*g*Cx*Sx + m*L**2*(m*L*theta_dot**2*Sx - d*x_dot)) + m*L*L*(1/D)*u
A3 = theta_dot
A4 = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*theta_dot**2*Sx - d*x_dot)) - m*L*Cx*(1/D)*u

q = sy.Matrix([x, x_dot, theta, theta_dot])
A = sy.Matrix([A1, A2, A3, A4])
u_vec = sy.Matrix([u])
A_lin_temp = A.jacobian(q)
B_lin_temp = A.jacobian(u_vec)

# fixed point 1. => 실제로는 이것 사용
# x_dot = 0 / theta = 0 / theta_dot = 0
A_lin_case1 = A_lin_temp.subs([(x_dot, 0), (theta, 0), (theta_dot, 0)])
B_lin_case1 = B_lin_temp.subs([(x_dot, 0), (theta, 0), (theta_dot, 0)])
print("A_lin_case1")
print(sy.simplify(A_lin_case1))
print("B_lin_case1")
print(sy.simplify(B_lin_case1))
print()

# fixed point 2. 
# x_dot = 0 / theta = sy.pi / theta_dot = 0
A_lin_case2 = A_lin_temp.subs([(x_dot, 0), (theta, sy.pi), (theta_dot, 0)])
B_lin_case2 = B_lin_temp.subs([(x_dot, 0), (theta, sy.pi), (theta_dot, 0)])
print("A_lin_case2")
print(sy.simplify(A_lin_case2))
print("B_lin_case2")
print(sy.simplify(B_lin_case2))
print()

#########################
# EOM from derive_ip.py #
#########################

# dx = z[1]
# ax = 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + g*m*np.sin(2*theta)/2 + u)/(M + m*np.sin(theta)**2)
# omega = z[3]
# alpha = -(1.0*g*(M + m)*np.sin(theta) + 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + u)*np.cos(theta))/(L*(M + m*np.sin(theta)**2))

A1 = x_dot
A2 = 1*(1*L*m*theta_dot**2*sy.sin(theta) - d*x_dot + g*m*sy.sin(2*theta)/2 + u)/(M + m*sy.sin(theta)**2)
A3 = theta_dot
A4 = -(1*g*(M + m)*sy.sin(theta) + 1*(1*L*m*theta_dot**2*sy.sin(theta) - d*x_dot + u)*sy.cos(theta))/(L*(M + m*sy.sin(theta)**2))

q = sy.Matrix([x, x_dot, theta, theta_dot])
A = sy.Matrix([A1, A2, A3, A4])
u_vec = sy.Matrix([u])
A_lin_temp = A.jacobian(q)
B_lin_temp = A.jacobian(u_vec)

# fixed point 1. => 실제로는 이것 사용
# x_dot = 0 / theta = 0 / theta_dot = 0
A_lin_case1 = A_lin_temp.subs([(x_dot, 0), (theta, 0), (theta_dot, 0)])
B_lin_case1 = B_lin_temp.subs([(x_dot, 0), (theta, 0), (theta_dot, 0)])
print("A_lin_case1")
print(sy.simplify(A_lin_case1))
print("B_lin_case1")
print(sy.simplify(B_lin_case1))
print()

# fixed point 2. 
# x_dot = 0 / theta = sy.pi / theta_dot = 0
A_lin_case2 = A_lin_temp.subs([(x_dot, 0), (theta, sy.pi), (theta_dot, 0)])
B_lin_case2 = B_lin_temp.subs([(x_dot, 0), (theta, sy.pi), (theta_dot, 0)])
print("A_lin_case2")
print(sy.simplify(A_lin_case2))
print("B_lin_case2")
print(sy.simplify(B_lin_case2))
