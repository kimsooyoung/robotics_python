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

x, x_d, x_dd = sy.symbols('x x_d x_dd', real=True)

# ex1
f1 = sy.sin(x)
df1dx = sy.diff(f1, x) * x_d
print(df1dx)

# ex2
f2 = x * x_d
df2fx = sy.diff(f2, x) * x_d + sy.diff(f2, x_d) * x_dd
print(df2fx)
