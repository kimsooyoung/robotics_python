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
from scipy.integrate import solve_ivp


class Param:

    def __init__(self):
        self.g = 9.81
        self.m = 1
        self.c = 0
        self.e = 0.8

        self.pause = 0.01
        self.fps = 10


# trigger function
def contact(t, z, m, g, c):
    x, xdot, y, ydot = z

    # y가 0일 때가 trigger 조건
    return y - 0


def projectile(t, z, m, g, c):

    x, x_dot, y, y_dot = z
    v = np.sqrt(x_dot**2 + y_dot**2)

    drag_x = c * v * x_dot
    drag_y = c * v * y_dot

    ax = 0 - (drag_x / m)
    ay = -g - (drag_y / m)

    return x_dot, ax, y_dot, ay


def one_bounce(t0, z0, params):

    t_end = t0 + 5
    contact.terminal = True
    # 위에서 아래로 내려갈 때만 event 발생
    contact.direction = -1

    sol = solve_ivp(
        fun=projectile, t_span=(t0, t_end), y0=z0, method='RK45',
        t_eval=np.linspace(t0, t_end, 1001), dense_output=True,
        events=contact, args=(params.m, params.g, params.c)
    )

    # sol.y => (4, 1001) / [x, x_dot, y, y_dot]
    # [[ 0.    0.    0.   ...  0.    0.    0.  ]
    # [ 0.    0.    0.   ...  0.    0.    0.  ]
    # [ 2.    2.05  2.1  ... 51.9  51.95 52.  ]
    # [10.   10.   10.   ... 10.   10.   10.  ]]

    t = sol.t

    m, n = sol.y.shape
    z = np.zeros((n, m))
    z = sol.y.T

    z[-1, 3] *= -params.e

    return t, z


def simulation(t0, t_end, z0, params):

    t = np.array([t0])
    z = np.zeros((1, 4))

    while t0 < t_end:
        t_temp, z_temp = one_bounce(t0, z0, params)

        t0 = t_temp[-1]
        z0 = z_temp[-1]

        z = np.concatenate((z, z_temp), axis=0)
        t = np.concatenate((t, t_temp), axis=0)

    return t, z


def animate(t, z, parms):
    # interpolationds
    # m, n을 fps에 따라서 x, n으로 늘리는 작업이다.
    t_interp = np.arange(t[0], t[len(t)-1], 1 / parms.fps)

    [m, n] = np.shape(z)
    shape = (len(t_interp), n)
    z_interp = np.zeros(shape)

    for i in range(0, n-1):
        f = interpolate.interp1d(t, z[:, i])
        z_interp[:, i] = f(t_interp)

    # 실제 그림 그리기
    for i in range(0, len(t_interp)):
        prj, = plt.plot(
            z_interp[i, 0], z_interp[i, 2],
            color='red', marker='o'
        )
        plt.plot([-2, 2], [0, 0], linewidth=2, color='black')

        plt.xlim(min(z[:, 0] - 1), max(z[:, 0] + 1))
        plt.ylim(min(z[:, 2] - 1), max(z[:, 2] + 1))

        plt.pause(parms.pause)

        if (i != len(t_interp)):
            prj.remove()

    plt.close()


def plot_result(t, z):

    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(t, z[:, 2], 'r')
    plt.ylabel('y')

    plt.subplot(2, 1, 2)
    plt.plot(t, z[:, 3], 'r')
    plt.ylabel('ydot')
    plt.xlabel('time')

    plt.show(block=False)
    plt.pause(3)
    plt.close()


if __name__ == '__main__':

    params = Param()

    # define initial cond
    # 초기 위로 올라가는 속도 10
    x0, x0dot, y0, y0dot = (0, 0, 2, 10)
    t0, t_end = (0, 10)

    # initial state
    z0 = np.array([x0, x0dot, y0, y0dot])

    t, z = simulation(t0, t_end, z0, params)
    animate(t, z, params)
    plot_result(t, z)
