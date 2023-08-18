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
from scipy.integrate import odeint


def free_fall(z, t, g):
    y, y_d = z

    y_dd = -g

    return [y_d, y_dd]


t_0, t_end, N = 0, 3, 100
ts = np.linspace(t_0, t_end, N)

result = odeint(free_fall, [0, 5], ts, args=(9.8,))

plt.plot(ts, result[:, 0], 'b', label='y(t)')
plt.plot(ts, result[:, 1], 'g', label="y\'(t)")

plt.legend(loc='best')
plt.show()
