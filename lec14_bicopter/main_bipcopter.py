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


class Parameters:

    def __init__(self):
        self.m = 1
        self.g = 9.81
        self.l = 0.2
        self.r = 0.05
        self.I = self.m * self.l**2 / 12

        self.pause = 0.01
        self.fps = 30


def controller(m, g, l, r, I):
    us = m * g + 0.01
    ud = 0.0

    return us, ud


def bicopter_dynamics(z, t, m, g, l, r, I):
    x, y, phi, x_d, y_d, phi_d = z
    us, ud = controller(m, g, l, r, I)

    x_dd = -(us) * np.sin(phi) / m
    y_dd = -g + us * np.cos(phi) / m
    phi_dd = (l * ud) / (2 * I)

    return x_d, y_d, phi_d, x_dd, y_dd, phi_dd


def animate(t, z, parms):
    # interpolation
    t_interp = np.arange(t[0], t[len(t) - 1], 1 / parms.fps)
    [m, n] = np.shape(z)
    shape = (len(t_interp), n)
    z_interp = np.zeros(shape)

    for i in range(0, n - 1):
        f = interpolate.interp1d(t, z[:, i])
        z_interp[:, i] = f(t_interp)

    l = parms.l
    r = parms.r

    xxyy = 1

    # plot
    for i in range(0, len(t_interp)):
        x = z_interp[i, 0]
        y = z_interp[i, 1]
        phi = z_interp[i, 2]

        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        middle = np.array([x, y])

        drone_left = np.add(middle, R.dot(np.array([-0.5 * l, 0])))
        axle_left = np.add(middle, R.dot(np.array([-0.5 * l, 0.1])))
        prop_left1 = np.add(
            middle,
            np.add(R.dot(np.array([-0.5 * l, 0.05])), R.dot(np.array([0.5 * r, 0.0]))),
        )
        prop_left2 = np.add(
            middle,
            np.add(R.dot(np.array([-0.5 * l, 0.05])), R.dot(np.array([-0.5 * r, 0.0]))),
        )

        drone_right = np.add(middle, R.dot(np.array([0.5 * l, 0])))
        axle_right = np.add(middle, R.dot(np.array([0.5 * l, 0.1])))
        prop_right1 = np.add(
            middle,
            np.add(R.dot(np.array([0.5 * l, 0.05])), R.dot(np.array([0.5 * r, 0.0]))),
        )
        prop_right2 = np.add(
            middle,
            np.add(R.dot(np.array([0.5 * l, 0.05])), R.dot(np.array([-0.5 * r, 0.0]))),
        )

        (drone,) = plt.plot(
            [drone_left[0], drone_right[0]],
            [drone_left[1], drone_right[1]],
            linewidth=5,
            color='red',
        )
        (prop_left_stand,) = plt.plot(
            [drone_left[0], axle_left[0]],
            [drone_left[1], axle_left[1]],
            linewidth=5,
            color='green',
        )
        (prop_left,) = plt.plot(
            [prop_left1[0], prop_left2[0]],
            [prop_left1[1], prop_left2[1]],
            linewidth=5,
            color='blue',
        )
        (prop_right_stand,) = plt.plot(
            [drone_right[0], axle_right[0]],
            [drone_right[1], axle_right[1]],
            linewidth=5,
            color='green',
        )
        (prop_right,) = plt.plot(
            [prop_right1[0], prop_right2[0]],
            [prop_right1[1], prop_right2[1]],
            linewidth=5,
            color='blue',
        )

        (endEff,) = plt.plot(x, y, color='black', marker='o', markersize=2)

        plt.xlim(-xxyy - 0.1, xxyy + 0.1)
        plt.ylim(-xxyy - 0.1, xxyy + 0.1)
        plt.gca().set_aspect('equal')

        plt.pause(parms.pause)
        drone.remove()
        prop_left_stand.remove()
        prop_left.remove()
        prop_right_stand.remove()
        prop_right.remove()

    plt.close()


if __name__ == '__main__':
    params = Parameters()
    args = params.m, params.g, params.l, params.r, params.I

    t, tend, N = 0, 5, 100
    ts = np.linspace(t, tend, N)

    x, y, phi, x_d, y_d, phi_d = 0, 0, 0, 0, 0, 0
    z0 = x, y, phi, x_d, y_d, phi_d
    tau = np.zeros((N, 2))
    z = np.zeros((N, 6))
    z[0] = z0

    for i in range(N - 1):
        t_temp = np.array([ts[i], ts[i + 1]])
        result = odeint(bicopter_dynamics, z0, t_temp, args)

        z0 = result[-1]
        z[i + 1] = z0

    animate(ts, z, params)
