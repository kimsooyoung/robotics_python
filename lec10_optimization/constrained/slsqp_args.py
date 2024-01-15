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


def cost(x):
    x1, x2, x3, x4, x5 = x
    return x1**2 + x2**2 + x3**2 + x4**2 + x5**2


limits = opt.Bounds(
    [0.3, -inf, -inf, -inf, -inf],
    [inf,  inf,    5,  inf,  inf]
)

ineq_const = {
    'type': 'ineq',
    'fun': lambda x: np.array([5 - x[3] ** 2 - x[4] ** 2]),
}

eq_const = {
    'type': 'eq',
    'fun': lambda x: np.array([
        5 - x[0] - x[1] - x[2],
        2 - x[2]**2 - x[3]
    ]),
}

x0 = np.array([1, 1, 1, 1, 1])
res = opt.minimize(
    cost, x0, method='SLSQP', constraints=[eq_const, ineq_const],
    options={'ftol': 1e-9, 'disp': True}, bounds=limits
)
print(res.x)
