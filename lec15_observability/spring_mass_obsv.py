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

import control
import numpy as np


class Paraemters:

    def __init__(self):
        self.m1, self.m2 = 1, 1
        self.k1, self.k2 = 2, 3


def dynamics(m1, m2, k1, k2):
    A = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-(k1 / m1 + k2 / m1), k2 / m1, 0, 0],
            [k2 / m2, -k2 / m2, 0, 0],
        ]
    )

    B = np.array([[0, 0], [0, 0], [-1 / m1, 0], [1 / m2, 1 / m2]])

    # observe velocity
    C = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])

    return A, B, C


if __name__ == '__main__':
    params = Paraemters()
    m1, m2, k1, k2 = params.m1, params.m2, params.k1, params.k2

    A, B, C = dynamics(m1, m2, k1, k2)

    # 1. compute eigenvalues of unestimated system
    eigVal, eigVec = np.linalg.eig(A)
    print('eig-vals (unestimated)')
    print(eigVal, '\n')

    # 2. compute observability of the system (2 ways)
    # 2.1. compute observability matrix
    Ob = control.obsv(A, C)
    print('control.obsv(A,C)')
    print(Ob)
    # print(f'rank={np.linalg.matrix_rank(Ob)}')

    # 2.2. compute observability matrix using transpose of controllability matrix
    Ob_trans = control.ctrb(A.T, C.T)
    print('control.ctrb(A.T, C.T)')
    print(Ob_trans.T)

    # 3. observability stability
    rank = np.linalg.matrix_rank(Ob)
    print('Rank of Ob')
    print(rank)

    # 4. pole replacement for stable observability
    p = np.array([-0.5, -0.6, -0.65, -6])
    L_trans = control.place(A.T, C.T, p)
    L = L_trans.T
    print('L')
    print(L)

    # 5. check new poles again
    new_A = A - L @ C
    eigVal, eigVec = np.linalg.eig(new_A)
    print('eig-vals (controlled)')
    print(eigVal)
