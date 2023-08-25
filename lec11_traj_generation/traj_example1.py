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

# q1(0) = 0
# q1(1) = 0.5
# q2(1) = 0.5
# q2(3) = 1
# q1_d(0) = 0
# q1_d(1) = 0.2
# q2_d(1) = 0.2
# q2_d(3) = 0

A = np.matrix('1 0 0 0 0 0 0 0; \
           1 1 1 1 0 0 0 0; \
           0 0 0 0 1 1 1 1; \
           0 0 0 0 1 3 9 27; \
           0 1 0 0 0 0 0 0 ; \
           0 1 2 3 0 0 0 0 ; \
           0 0 0 0 0 1 2 3 ; \
           0 0 0 0 0 1 6 27')

b = np.matrix('0; 0.5; 0.5; 1; 0; 0.2; 0.2; 0')
x = A.getI()*b
print(x)

# this doesn't work
# a10, a11, a12, a13 = x[:4,0]
# a20, a21, a22, a23 = x[4:,0]

a10 = x[0, 0]
a11 = x[1, 0]
a12 = x[2, 0]
a13 = x[3, 0]
a20 = x[4, 0]
a21 = x[5, 0]
a22 = x[6, 0]
a23 = x[7, 0]

t0, t1, t2 = 0, 1, 3

tt1 = np.linspace(t0, t1, 101)
tt2 = np.linspace(t1, t2, 101)

q1 = a10 + a11*tt1 + a12*tt1**2 + a13*tt1**3
q2 = a20 + a21*tt2 + a22*tt2**2 + a23*tt2**3

q1_d = a11 + 2*a12*tt1 + 3*a13*tt1**2
q2_d = a21 + 2*a22*tt2 + 3*a23*tt2**2

q1_dd = 2*a12 + 6*a13*tt1
q2_dd = 2*a22 + 6*a23*tt2

plt.figure(1)

plt.subplot(3, 1, 1)
plt.plot(tt1, q1, 'b-')
plt.plot(tt2, q2, 'r--')
plt.ylabel("q")

plt.subplot(3, 1, 2)
plt.plot(tt1, q1_d, 'b-')
plt.plot(tt2, q2_d, 'r--')
plt.ylabel("q_d")

plt.subplot(3, 1, 3)
plt.plot(tt1, q1_dd, 'b-')
plt.plot(tt2, q2_dd, 'r--')
plt.ylabel("q_dd")

plt.show()
