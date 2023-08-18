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

import scipy.optimize as opt


def cost(param):
    x1, x2 = param

    return 100 * (x2 - x1)**2 + (1 - x1)**2


initial_val = [0, 0]

# result = opt.minimize(cost, initial_val, method='BFGS')
result = opt.minimize(cost, initial_val, method='CG')
print(result)
print(result.x)
