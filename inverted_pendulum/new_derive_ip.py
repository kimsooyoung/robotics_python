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
# L, I : pendulum length, inertia 
# g : gravity
# d : damping
m, M, L, I = sy.symbols('m M L I', real=True)
g, d = sy.symbols('g d', real=True)

######################
# Step 1. kinematics #
######################

pi = sy.pi
cos = sy.cos
sin = sy.sin

q = sy.Matrix([x, theta])
q_d = sy.Matrix([x_dot, theta_dot])

# floating base => x, y, 1
angle1 = theta - pi/2
H_01 = sy.Matrix([
    [cos(angle1), -sin(angle1), x],
    [sin(angle1), cos(angle1), 0],
    [0, 0, 1]
])

X_mass = sy.Matrix([L, 0, 1])
Xm = H_01 * X_mass
m_xy = sy.Matrix([Xm[0], Xm[1]])
v_xy = m_xy.jacobian(q) * q_d

####################
## Step 2. Energy ##
####################

T = 0.5 * M * x_dot**2 + \
    0.5 * m * v_xy.dot(v_xy) + \
    0.5 * I * theta_dot**2
V = m * g * Xm[1]
L = T - V

######################
# Step 3. E-L Method #
######################

dL_dq_d = []
dt_dL_dq_d = []
dL_dq = []
q_dd = sy.Matrix([x_ddot, theta_ddot])

EOM = []
Q = [u - d*x_dot, 0]

for i in range(len(q_dd)):
    dL_dq_d.append(sy.diff(L, q_d[i]))
    temp = 0
    for j in range(len(q_dd)):
        temp += sy.diff(dL_dq_d[i], q[j]) * q_d[j] + \
                sy.diff(dL_dq_d[i], q_d[j]) * q_dd[j]
    dt_dL_dq_d.append(temp)
    dL_dq.append(sy.diff(L, q[i]))
    EOM.append(dt_dL_dq_d[i] - dL_dq[i] - Q[i])

EOM = sy.Matrix(EOM)
print("Normal EOM")
print('EOM[0] = ', sy.simplify(EOM[0]))
print('EOM[1] = ', sy.simplify(EOM[1]))
print()

# Ax = b
A = EOM.jacobian(q_dd)
b_temp = []

for i in range(len(q_dd)):
    b_temp.append(-1 * EOM[i].subs([(x_ddot, 0), (theta_ddot, 0)]))
b = sy.Matrix([b_temp[0], b_temp[1]])

#######################
# Step 4. State-Space #
#######################

EOM_ss = A.inv() * b
print("State Equation")
print('EOM_ss[0] = ', sy.simplify(EOM_ss[0]))
print('EOM_ss[1] = ', sy.simplify(EOM_ss[1]))
print()

A_ss1 = x_dot
A_ss2 = EOM_ss[0]
A_ss3 = theta_dot
A_ss4 = EOM_ss[1]

A_temp = sy.Matrix([A_ss1, A_ss2, A_ss3, A_ss4])
q_ss = sy.Matrix([x, x_dot, theta, theta_dot])
u_ss = sy.Matrix([u])

A_ss = A_temp.jacobian(q_ss)
B_ss = A_temp.jacobian(u_ss)

#########################
# Step 5. Linearization #
#########################

# fixed point 1.
# x_dot = 0 / theta = 0 / theta_dot = 0
A_lin_case1 = A_ss.subs([(x_dot, 0), (theta, 0), (theta_dot, 0)])
B_lin_case1 = B_ss.subs([(x_dot, 0), (theta, 0), (theta_dot, 0)])
print("A_lin_case1")
print(sy.simplify(A_lin_case1))
print("B_lin_case1")
print(sy.simplify(B_lin_case1))
print()

# fixed point 2. 
# x_dot = 0 / theta = sy.pi / theta_dot = 0
A_lin_case2 = A_ss.subs([(x_dot, 0), (theta, sy.pi), (theta_dot, 0)])
B_lin_case2 = B_ss.subs([(x_dot, 0), (theta, sy.pi), (theta_dot, 0)])
print("A_lin_case2")
print(sy.simplify(A_lin_case2))
print("B_lin_case2")
print(sy.simplify(B_lin_case2))
