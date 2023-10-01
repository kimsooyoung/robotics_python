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

import matplotlib.pyplot as plt
import numpy as np

t0, tf = 0, 1
q0, qf = 10, 20

t = np.linspace(t0, tf, 101)

a0 = q0*tf/(-t0 + tf) - qf*t0/(-t0 + tf)
a1 = -q0/(-t0 + tf) + qf/(-t0 + tf)

q_t = a0 + a1*t
qdot_t = a1 * np.ones((len(t), 1))

plt.figure(1)

plt.subplot(2, 1, 1)
plt.plot(t, q_t)
plt.ylabel('q')

plt.subplot(2, 1, 2)
plt.plot(t, qdot_t)
plt.ylabel('qdot')

plt.show()
