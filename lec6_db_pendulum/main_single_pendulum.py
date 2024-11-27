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
from scipy import interpolate

from scipy.integrate import odeint


class parameters:

    def __init__(self):
        self.m = 1
        self.I = 0.1
        self.c = 0.5
        self.l = 1
        self.g = 9.81
        self.pause = 0.02
        self.fps = 20


def cos(angle):
    return np.cos(angle)


def sin(angle):
    return np.sin(angle)


def interpolation(t, z, params):

    # interpolation
    t_interp = np.arange(t[0], t[len(t)-1], 1/params.fps)
    [rows, cols] = np.shape(z)
    z_interp = np.zeros((len(t_interp), cols))

    for i in range(0, cols):
        f = interpolate.interp1d(t, z[:, i])
        z_interp[:, i] = f(t_interp)

    return t_interp, z_interp


def animate(t_interp, z_interp, params):

    l = params.l
    c = params.c

    # plot
    for i in range(0, len(t_interp)):

        theta = z_interp[i, 0]

        O = np.array([0, 0])
        P = np.array([l*sin(theta), -l*cos(theta)])

        # COM Point
        G = np.array([c*sin(theta), -c*cos(theta)])

        pend, = plt.plot(
            [O[0], P[0]], [O[1], P[1]], linewidth=5, color='red'
        )

        com, = plt.plot(
            G[0], G[1], color='black', marker='o', markersize=10
        )

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal')

        plt.pause(params.pause)

        if (i < len(t_interp)-1):
            pend.remove()
            com.remove()

    plt.show(block=False)
    plt.pause(1)
    plt.close()

    # result plotting
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t, z[:, 0], color='red', label='theta1')
    plt.plot(t, z[:, 2], color='blue', label='theta2')
    plt.ylabel('angle')
    plt.legend(loc='upper left')

    plt.subplot(2, 1, 2)
    plt.plot(t, z[:, 1], color='red', label='omega1')
    plt.plot(t, z[:, 3], color='blue', label='omega2')
    plt.xlabel('t')
    plt.ylabel('angular rate')
    plt.legend(loc='lower left')

    plt.show()


def single_pendulum(z0, t, m, I, c, l, g):

    theta, omega = z0

    M = 1.0*I + 0.5*m*(2*c**2*sin(theta)**2 + 2*c**2*cos(theta)**2)
    C = 0
    G = c*g*m*sin(theta)

    theta_dd = -(C + G) / M

    return [omega, theta_dd]


if __name__ == '__main__':

    params = parameters()

    t = np.linspace(0, 10, 500)

    # initlal state
    z0 = np.array([np.pi, 0.001])
    all_params = (
        params.m, params.I,
        params.c, params.l,
        params.g
    )
    z = odeint(single_pendulum, z0, t, args=all_params)

    t_interp, z_interp = interpolation(t, z, params)
    animate(t_interp, z_interp, params)
