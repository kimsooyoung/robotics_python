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
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint


class Parameters:

    def __init__(self):
        self.m1, self.m2 = 1, 1
        self.k1, self.k2 = 2, 3

        self.Q = np.eye(4)
        self.R = np.eye(2)

        self.pause = 0.01


def dynamisc(m1, m2, k1, k2):
    A = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-(k1 / m1 + k2 / m1), k2 / m1, 0, 0],
            [k2 / m2, -k2 / m2, 0, 0],
        ]
    )

    B = np.array([[0, 0], [0, 0], [-1 / m1, 0], [1 / m2, 1 / m2]])

    return A, B


def get_control(x, K):
    return -K @ x


def spring_mass_linear_equ(x, t, m1, m2, k1, k2, K):
    A, B = dynamisc(m1, m2, k1, k2)

    # uncontrolled
    u = np.array([0,0])

    # Linear control
    # u = get_control(x, K)

    return A @ x + B @ u


def plot(result, u, ts):
    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(ts, result[:, 0], 'r-.', label='x1')
    plt.plot(ts, result[:, 1], 'b', label='x2')
    plt.title('Position and Velocity vs. time')
    plt.ylabel('position')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(ts, result[:, 2], 'r-.', label='v1')
    plt.plot(ts, result[:, 3], 'b', label='v2')
    plt.ylabel('velocity')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(ts, u[:, 0], 'b')
    plt.title('U1 and U2 vs. time')
    plt.ylabel('u1')

    plt.subplot(2, 1, 2)
    plt.plot(ts, u[:, 1], 'b')
    plt.ylabel('u2')
    plt.show()


if __name__ == '__main__':
    params = Parameters()

    m1, m2, k1, k2 = params.m1, params.m2, params.k1, params.k2
    Q, R = params.Q, params.R

    A, B = dynamisc(m1, m2, k1, k2)

    # Pole placement
    p = [-1,-2,-3,-4]
    # p = [-10,-20,-30,-40]
    K = control.place(A,B,p)
    eigVal_p, eigVec_p = np.linalg.eig(A - B@K)
    print(f'new eigVal, eigVec: \n {eigVal_p} \n {eigVec_p}')

    # # LQR
    # K, _, E = control.lqr(A, B, Q, R)
    # print(f'K = {K}')
    # print(f'E = {E}')

    t0, tend, N = 0, 10, 101
    ts = np.linspace(t0, tend, N)

    z0 = np.array([0.5, 0, 0, 0])
    z = np.zeros((N, 4))
    u = np.zeros((N, 2))
    z[0] = z0

    for i in range(N - 1):
        t_temp = np.array([ts[i], ts[i + 1]])
        z_temp = odeint(spring_mass_linear_equ, z0, t_temp, args=(m1, m2, k1, k2, K))
        u[i] = get_control(z0, K)

        z0 = z_temp[-1]
        z[i + 1] = z0

    plot(z, u, ts)
