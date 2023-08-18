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
from scipy.optimize import fsolve


def func(x):
    return x**2 - x - 2


# case 1
root = fsolve(func, 3)
# case 2
# root = fsolve(func, 0)

x = np.arange(-6, 6, 0.1)   # start,stop,step
y = func(x)

plt.figure(1)

plt.plot(x, y)
plt.plot([-6, 6], [0, 0], color='black', linewidth=1)
plt.plot(root, 0, color='green', marker='o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of func')

plt.grid()
plt.show(block=False)
plt.pause(5)
plt.close()
