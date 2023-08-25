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


class Parameters():

    def __init__(self):
        # system parameters
        self.m = 1
        self.c = 1
        self.k = 1

        # control gains
        self.kp = 0.1
        self.kd = -self.c + 2 * np.sqrt((self.k + self.kp) * self.m)


def smd_rhs(z, t, m, c, k, kp, kd):

    x, xdot = z[0], z[1]

    # w/o control
    # xdotdot = -(c*xdot + k*x)/m

    # with pd control - feedback linearization
    xdotdot = -((c+kd)*xdot + (k+kp)*x)/m

    return [xdot, xdotdot]


def plot(t, z):

    plt.figure(1)

    plt.plot(t, z[:, 0])

    plt.xlabel('t')
    plt.ylabel('position')
    plt.title('Plot of position vs time')

    plt.show()


if __name__ == '__main__':

    params = Parameters()
    m, c, k, kp, kd = params.m, params.c, params.k, params.kp, params.kd

    # let's assume mass object initially located in 0.5 point
    x0, xdot0 = 0.5, 0
    t0, tend = 0, 20

    t = np.linspace(t0, tend)
    z0 = np.array([x0, xdot0])

    result = odeint(smd_rhs, z0, t, args=(m, c, k, kp, kd))

    plot(t, result)
