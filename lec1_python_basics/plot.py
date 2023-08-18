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

t = np.linspace(0, 6, 50)
y = np.sin(t)

plt.figure(1)
# b => blue, o => circle
plt.plot(t, y, 'bo')
plt.xlabel('t')
plt.ylabel('sin(t)')
plt.show(block=False)
plt.pause(5)
plt.close()
