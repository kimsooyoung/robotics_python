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

import numpy as np
import matplotlib.pyplot as plt

# θ_dot_1(1) = θ_dot_2(1) = 0.2를 버리고
# θ_dot_1(1) = θ_dot_2(1) / θ_dotdot_2(1) = θ_dotdot_1(1)를 추가하였다.

A = np.matrix('1 0 0 0 0 0 0 0; \
           1 1 1 1 0 0 0 0; \
           0 0 0 0 1 1 1 1; \
           0 0 0 0 1 3 9 27; \
           0 1 0 0 0 0 0 0 ; \
           0 1 2 3 0 -1 -2 -3; \
           0 0 1 3 0 0 -1 -3; \
           0 0 0 0 0 1 6 27')

b = np.matrix('0; 0.5; 0.5; 1; 0; 0; 0; 0')
x = A.getI()*b
# print(x)

a10 = x[0, 0]
a11 = x[1, 0]
a12 = x[2, 0]
a13 = x[3, 0]
a20 = x[4, 0]
a21 = x[5, 0]
a22 = x[6, 0]
a23 = x[7, 0]

t1 = np.linspace(0, 1, 101)
t2 = np.linspace(1, 3, 101)

q1 = a10 + a11*t1 + a12*t1**2 + a13*t1**3
q2 = a20 + a21*t2 + a22*t2**2 + a23*t2**3

q1dot = a11 + 2*a12*t1 + 3*a13*t1**2
q2dot = a21 + 2*a22*t2 + 3*a23*t2**2

q1ddot = 2*a12 + 6*a13*t1
q2ddot = 2*a22 + 6*a23*t2

plt.figure(1)

plt.subplot(3, 1, 1)
plt.plot(t1, q1, 'b-')
plt.plot(t2, q2, 'r--')
plt.ylabel('q')

plt.subplot(3, 1, 2)
plt.plot(t1, q1dot, 'b-')
plt.plot(t2, q2dot, 'r--')

plt.ylabel('qdot')
plt.subplot(3, 1, 3)
plt.plot(t1, q1ddot, 'b-')
plt.plot(t2, q2ddot, 'r--')
plt.ylabel('qddot')
plt.show()
