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
from scipy.integrate import solve_ivp


# differential equation
def dydt(t, y):
    return -y + 2*np.sin(t)


def event(t, y):
    return y[0] - 1


event.terminal = False
event.direction = -1

ts = np.linspace(0, 10, 100)
initial_state = (1, 1)
sol = solve_ivp(dydt, t_span=(0, 10), y0=[10], t_eval=ts, events=event)

t = sol.t
y = sol.y[0]
events = sol.t_events[0]

plt.plot(t, y)
for point in events:
    plt.plot(point, 1, color='green', marker='o', markersize=10)
plt.show()
