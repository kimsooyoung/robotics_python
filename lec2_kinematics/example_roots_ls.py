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

from scipy.optimize import least_squares


# symbolic answer => -1, +2
def func(x):
    return x**2 - x - 2


# state가 2개 이상이 되면 하나의 tuple로 묶어서 전달해야 함
def multi_var_func(var):
    x, y = var
    return x**2 - x - 2 + y**2 - y - 2


# parameter는 튜플로 묶이지 않고 각각 전달됨
def multi_var_func_w_params(var, radius, nothing):
    x, y = var
    return (x - 2)**2 + (y - 2)**2 - radius**2


res1 = least_squares(func, 0, bounds=((-1), (0)))
print(res1.x)

res2 = least_squares(multi_var_func, (0, 0), bounds=((-1), (0)))
print(res2.x)

radius = 3
nothing = 12.34
res3 = least_squares(
    multi_var_func_w_params, (0, 0),
    bounds=((-1, -1), (2, 2)),
    args=(radius, nothing)
)
print(res3.x)
