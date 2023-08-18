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
import scipy.optimize as opt

inf = np.inf


def cost(param):
    x1, x2, x3, x4, x5 = param
    return x1**2 + x2**2 + x3**2 + x4**2 + x5**2


limits = opt.Bounds(
    [0.3, -inf, -inf, -inf, -inf],
    [inf,  inf,    5,  inf,  inf]
)


linear_const = opt.LinearConstraint([1, 1, 1, 0, 0], [5], [5])
nonlinear_const = opt.NonlinearConstraint(
    lambda x: [5 - x[3]**2 - x[4]**2, 2 - x[2]**2 - x[3]],
    [0, 0], [np.inf, 0]
)

x0 = np.array([1, 1, 1, 2, 1])
res = opt.minimize(
    cost, x0, method='trust-constr',
    constraints=[linear_const, nonlinear_const],
    options={'xtol': 1e-9, 'verbose': 1},
    bounds=limits
)
print(res.x)
