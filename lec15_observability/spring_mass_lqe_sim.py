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


class Parameter:

    def __init__(self):
        self.m1, self.m2 = 1, 1
        self.k1, self.k2 = 2, 3

        self.A = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [-(self.k1 / self.m1 + self.k2 / self.m1), self.k2 / self.m1, 0, 0],
                [self.k2 / self.m2, -self.k2 / self.m2, 0, 0],
            ]
        )

        self.C = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])

        self.G = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

        self.Qe = np.diag([0.1, 0.1])
        self.Re = np.diag([0.1, 0.1])

        # self.Qe = np.diag([2,3])
        # self.Re = np.diag([0.5,0.5])


def spring_mass_dynamics(x, t, A, C, G, L, p_noise1, p_noise2, m_noise1, m_noise2):
    O_44 = np.zeros((4, 4))
    O_42 = np.zeros((4, 2))

    Gbig = np.block([[O_42], [G]])

    Lbig = np.block([[O_42], [L]])

    Abig = np.block([[A, O_44], [L @ C, A - L @ C]])

    # w: process noise, v: measurement noise
    w = np.array([p_noise1, p_noise2])
    v = np.array([m_noise1, m_noise2])

    xdot = Abig @ x + Gbig @ w + Lbig @ v

    return xdot


def plot(t, z):
    plt.figure(1)

    plt.subplot(2, 2, 1)
    plt.plot(t, z[:, 0], 'r-.')
    plt.plot(t, z[:, 4], 'b')
    plt.ylabel('position q1')
    plt.legend(['act', 'est'])

    plt.subplot(2, 2, 3)
    plt.plot(t, z[:, 1], 'r-.')
    plt.plot(t, z[:, 5], 'b')
    plt.legend(['act', 'est'])
    plt.ylabel('position q2')
    plt.xlabel('time t')

    plt.subplot(2, 2, 2)
    plt.plot(t, z[:, 2], 'r-.')
    plt.plot(t, z[:, 6], 'b')
    plt.ylabel('velocity q1dot ')
    plt.legend(['act', 'est'])

    plt.subplot(2, 2, 4)
    plt.plot(t, z[:, 3], 'r-.')
    plt.plot(t, z[:, 7], 'b')
    plt.ylabel('velocity q2dot ')
    plt.xlabel('time t')
    plt.legend(['act', 'est'])

    plt.show(block=False)
    plt.pause(10)
    plt.close()


if __name__ == '__main__':
    param = Parameter()
    m1, m2, k1, k2 = param.m1, param.m2, param.k1, param.k2
    A, C, G = param.A, param.C, param.G
    Qe, Re = param.Qe, param.Re

    # kalman gain
    L, P, E = control.lqe(A, G, C, Qe, Re)
    print('L\n', L)
    print('E\n', E)

    # process, measurement noise
    p_noise1_mean, p_noise1_std = 0, np.sqrt(Qe[0, 0])
    p_noise2_mean, p_noise2_std = 0, np.sqrt(Qe[1, 1])
    m_noise1_mean, m_noise1_std = 0, np.sqrt(Re[0, 0])
    m_noise2_mean, m_noise2_std = 0, np.sqrt(Re[1, 1])

    p_noise = np.random.random

    t0, tend = 0, 10
    ts = np.linspace(t0, tend, 1000)

    z_real = np.array([0.5, 0.5, 0.5, 0.5])
    # z_noisy = np.array([0.3, 0, 0, 0])
    z_noisy = np.array([0, 0, 0, 0])
    z0 = np.concatenate((z_real, z_noisy))

    z = np.zeros((len(ts), 8))
    z[0] = z0

    args = A, C, G, L

    for i in range(len(ts) - 1):
        p_noise1 = np.random.normal(p_noise1_mean, p_noise1_std)
        p_noise2 = np.random.normal(p_noise2_mean, p_noise2_std)
        m_noise1 = np.random.normal(m_noise1_mean, m_noise1_std)
        m_noise2 = np.random.normal(m_noise2_mean, m_noise2_std)

        total_args = args + (p_noise1, p_noise2, m_noise1, m_noise2)

        t_temp = np.array([ts[i], ts[i + 1]])

        result = odeint(spring_mass_dynamics, z0, t_temp, total_args)

        z0 = result[1]
        z[i + 1] = z0

    plot(ts, z)
