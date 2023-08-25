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

# q0 = a0 + a1*t0
# qf = a0 + a1*tf
#
# Ax = b

a0, a1 = sy.symbols('a0 a1')
q0, qf = sy.symbols('q0 qf')
t0, tf = sy.symbols('t0 tf')

A = sy.Matrix([
    [1, t0],
    [1, tf]
])

b = sy.Matrix([
    [q0],
    [qf]
])

x = A.inv() * b
print(f"a0 = {x[0]}")
print(f"a1 = {x[1]}")
