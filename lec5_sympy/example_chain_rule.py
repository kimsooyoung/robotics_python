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
f0 = sy.sin(x**2)

df0_fx = sy.diff(f0, x)
print(f'df0_fx : {df0_fx}')

f1 = x * x_d
df1_fx = sy.diff(f1, x) * x_d + sy.diff(f1, x_d) * x_dd
print(f'df1_fx : {df1_fx}')
