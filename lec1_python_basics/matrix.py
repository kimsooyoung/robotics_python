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

# 1. numpy matrix
A = np.array([[2, 4], [5, -6]])
print(f'A = \n{A}\n')


# rotation matrix generation func
def rot_mat(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])


def trans_vec(x, y):
    return np.array([
        [x],
        [y],
    ])


if __name__ == '__main__':

    # (2 X 2) dot (2 X 1) => (2 X 1)
    B = np.array([[2], [2]])
    D = A.dot(B)
    D_ = A @ B
    print(f'A dot B = \n{D}\n')
    print(f'A @ B = \n{D_}\n')

    # Transpose
    A_T = A.transpose()
    print(f'A_T = \n{A_T}\n')
    # or
    # print(np.transpose(A))

    # inverse
    inv_a = np.linalg.inv(A)
    print(f'inv_a = \n{inv_a}\n')

    # # err case
    # err_case = np.array([[1,0],[0,0]])
    # inv_a = np.linalg.inv(err_case)

    # Element-wise mult & Matrix mult
    print('inv_a * A =\n', inv_a * A)
    print('np.matmul(inv_a, A) =\n', np.matmul(inv_a, A))
    print('inv_a.dot(A) =\n', inv_a.dot(A))
    print('inv_a @ A =\n', inv_a @ A)
    print()

    theta = 30 * np.pi / 180
    print('Rotation Matrix = \n', rot_mat(theta))
    lx, ly = 0.5, 1.0
    print('Translation Vector = \n', trans_vec(lx, ly))

    # identity matrix
    print()
    print(np.identity(3))
    print(np.zeros((2, 4)))
