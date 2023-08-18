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


def func(x, y):
    return np.array([[x**2+y**2], [2*x+3*y+5]])


z = np.array([1, 2])
f = func(z[0], z[1])
epsilon = 1e-3

# J = ([
#     [df1/dx, df1/dy],
#     [df2/dx, df2/dy]
# ])
J = np.eye(2)

# x
dfdx = (func(z[0] + epsilon, z[1]) - func(z[0], z[1])) / epsilon
# dfdx.shape => 2 * 1 [[],[]]
J[0, 0] = dfdx[0, 0]
J[1, 0] = dfdx[1, 0]

dfdy = (func(z[0], z[1] + epsilon) - func(z[0], z[1])) / epsilon
J[0, 1] = dfdy[0, 0]
J[1, 1] = dfdy[1, 0]

print(J)
