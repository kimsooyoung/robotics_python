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

t0, tf = 0, 1
q0, qf = 10, 20

t = np.linspace(t0, tf, 101)

a0 = -q0*(10*t0**2*tf**3 - 5*t0*tf**4 + tf**5)/(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2 - 10*t0**2*tf**3 + 5*t0*tf**4 - tf**5) + qf*(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2)/(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2 - 10*t0**2*tf**3 + 5*t0*tf**4 - tf**5)
a1 = 30*q0*t0**2*tf**2/(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2 - 10*t0**2*tf**3 + 5*t0*tf**4 - tf**5) - 30*qf*t0**2*tf**2/(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2 - 10*t0**2*tf**3 + 5*t0*tf**4 - tf**5)
a2 = -q0*(30*t0**2*tf + 30*t0*tf**2)/(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2 - 10*t0**2*tf**3 + 5*t0*tf**4 - tf**5) + qf*(30*t0**2*tf + 30*t0*tf**2)/(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2 - 10*t0**2*tf**3 + 5*t0*tf**4 - tf**5)
a3 = q0*(10*t0**2 + 40*t0*tf + 10*tf**2)/(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2 - 10*t0**2*tf**3 + 5*t0*tf**4 - tf**5) - qf*(10*t0**2 + 40*t0*tf + 10*tf**2)/(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2 - 10*t0**2*tf**3 + 5*t0*tf**4 - tf**5)
a4 = -q0*(15*t0 + 15*tf)/(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2 - 10*t0**2*tf**3 + 5*t0*tf**4 - tf**5) + qf*(15*t0 + 15*tf)/(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2 - 10*t0**2*tf**3 + 5*t0*tf**4 - tf**5)
a5 = 6*q0/(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2 - 10*t0**2*tf**3 + 5*t0*tf**4 - tf**5) - 6*qf/(t0**5 - 5*t0**4*tf + 10*t0**3*tf**2 - 10*t0**2*tf**3 + 5*t0*tf**4 - tf**5)

q_t = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
qdot_t = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
qdotdot_t = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
qdotdotdot_t = 6*a3 + 24*a4*t + 60*a5*t**2

plt.figure(1)

plt.subplot(4, 1, 1)
plt.plot(t, q_t)
plt.ylabel('q')

plt.subplot(4, 1, 2)
plt.plot(t, qdot_t)
plt.ylabel('q_d')

plt.subplot(4, 1, 3)
plt.plot(t, qdotdot_t)
plt.ylabel('q_dd')

plt.subplot(4, 1, 4)
plt.plot(t, qdotdotdot_t)
plt.ylabel('q_ddd')

plt.show()
