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
from scipy import integrate


def f(t, r):
    x, y = r

    fx = np.cos(y)
    fy = np.sin(x)

    return fx, fy


def my_event(t, r):
    x, y = r
    return x - 1


my_event.terminal = False
my_event.direction = +1
sol = integrate.solve_ivp(
    f, t_span=(0, 10), y0=(1, 1),
    t_eval=np.linspace(0, 10, 100),
    events=my_event
)
events = sol.t_events[0]

t = sol.t
x, y = sol.y
plt.plot(t, x)

for point in events:
    plt.plot(point, 1, color='green', marker='o', markersize=10)

plt.axis('scaled')
plt.show()
