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

# stance phase eom

x, x_c, y = sy.symbols('x x_c y')
x_d, y_d = sy.symbols('x_d y_d')
m, g, k, l_0 = sy.symbols('m g k l_0')

# E-L euqation
l = sy.sqrt( (x-x_c)**2 + y**2)
# l = sy.symbols('l')
T = m/2 * (x_d**2 + y_d**2)
V = m*g*y + k/2 * (l - l_0) **2

L = T - V


# Lagrange equation
x_dd, y_dd = sy.symbols('x_dd y_dd', real=True)

q = sy.Matrix([x, y])
q_d = sy.Matrix([x_d, y_d])
q_dd = sy.Matrix([x_dd, y_dd])

dL_dq_d = []
dt_dL_dq_d = []
dL_dq = []
EOM = []

for i in range(len(q)):
    dL_dq_d.append(sy.diff(L, q_d[i]))
    temp = 0
    for j in range(len(q)):
        temp += sy.diff(dL_dq_d[i], q[j]) * q_d[j] + \
                sy.diff(dL_dq_d[i], q_d[j]) * q_dd[j]
    
    dt_dL_dq_d.append(temp)
    dL_dq.append(sy.diff(L, q[i]))

    EOM.append(dt_dL_dq_d[i] - dL_dq[i])
    
EOM = sy.Matrix([EOM[0],EOM[1]])

x_dd_solve = sy.solve(EOM[0], x_dd)
y_dd_solve = sy.solve(EOM[1], y_dd)
print(x_dd_solve)
print(y_dd_solve)