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

t = np.arange(0, 2 * np.pi, 0.1)
y = np.cos(t)
plt.plot(t, y)

for i in range(len(y)):
    temp,  = plt.plot(t[i], y[i], color='green', marker='o', markersize=10)
    plt.pause(0.005)
    temp.remove()

# plt.show()
plt.close()
