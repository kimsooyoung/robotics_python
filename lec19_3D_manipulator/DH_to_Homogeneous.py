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

import sympy as sy

theta, d, a, alpha = sy.symbols('theta d a alpha', real=True)

H_z_theta = sy.Matrix(
    [
        [sy.cos(theta), -sy.sin(theta), 0, 0],
        [sy.sin(theta), sy.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

H_z_d = sy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])

H_x_a = sy.Matrix([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

H_x_alpha = sy.Matrix(
    [
        [1, 0, 0, 0],
        [0, sy.cos(alpha), -sy.sin(alpha), 0],
        [0, sy.sin(alpha), sy.cos(alpha), 0],
        [0, 0, 0, 1],
    ]
)

H = H_z_theta * H_z_d * H_x_a * H_x_alpha

m, n = H.shape

for i in range(m):
    for j in range(n):
        print(f'H[i,j] = {H[i,j]}')
