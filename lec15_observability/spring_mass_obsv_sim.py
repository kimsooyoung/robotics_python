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

from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint


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


def spring_mass_dynamics(z, t, A, C, L):
    O_44 = np.zeros((4, 4))
    LC = L @ C

    # Abig = np.block([[A, O_44], [O_44, A]])

    Abig = np.block([
        [A,   O_44],
        [LC,  A-LC]
    ])

    return Abig @ z


def plot(t, x, parameters):
    plt.figure(1)

    plt.subplot(2, 2, 1)
    plt.plot(t, x[:, 0], 'r-.')
    plt.plot(t, x[:, 4], 'b')
    plt.ylabel('position q1')
    plt.legend(['act', 'est'])

    plt.subplot(2, 2, 3)
    plt.plot(t, x[:, 1], 'r-.')
    plt.plot(t, x[:, 5], 'b')
    plt.legend(['act', 'est'])
    plt.ylabel('position q2')
    plt.xlabel('time t')

    plt.subplot(2, 2, 2)
    plt.plot(t, x[:, 2], 'r-.')
    plt.plot(t, x[:, 6], 'b')
    plt.ylabel('velocity q1dot ')
    plt.legend(['act', 'est'])

    plt.subplot(2, 2, 4)
    plt.plot(t, x[:, 3], 'r-.')
    plt.plot(t, x[:, 7], 'b')
    plt.ylabel('velocity q2dot ')
    plt.xlabel('time t')
    plt.legend(['act', 'est'])

    plt.show(block=False)
    plt.pause(5)
    plt.close()


if __name__ == '__main__':
    params = Paraemters()
    m1, m2, k1, k2 = params.m1, params.m2, params.k1, params.k2

    t0, tend = 0, 10
    ts = np.linspace(t0, tend, 1000)

    # x0_real = np.array([0.5, 0.0, 0.0, 0.0])
    x0_real = np.array([0.5, 0.5, 0.5, 0.5])
    # x0_est = np.array([0.2, 0.0, 0.0, 0.0])
    x0_est = np.array([0.0, 0.0, 0.0, 0.0])
    x0 = np.concatenate((x0_real, x0_est))

    A, B, C = dynamics(m1, m2, k1, k2)
    L = np.array(
        [
            [0.53726568, 0.05436376],
            [0.05488936, 0.57214328],
            [4.73349624, -2.58818095],
            [-2.58429009, 3.01650376],
        ]
    )

    result = odeint(spring_mass_dynamics, x0, ts, args=(A, C, L))

    plot(ts, result, params)
