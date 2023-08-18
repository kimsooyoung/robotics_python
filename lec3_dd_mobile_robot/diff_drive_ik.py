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

import math

from matplotlib import pyplot as plt

import numpy as np
import dd_helper


class Parameters:

    def __init__(self):

        # 목적지에 도달해야 하는 offset
        # 목적지와 딱 맞게 가고 싶다면 px=0.0이어야 한다.
        self.px = 0.01
        self.py = 0.0
        self.Kp = 10

        # 로봇 반지름
        self.R = 0.1

        self.pause = 0.1
        self.fps = 5
        self.t_end = 10


def animate(params, t_interp, z_interp, z_dot, err):

    R = params.R
    phi = np.arange(0, 2*np.pi, 0.25)

    x_circle = R*np.cos(phi)
    y_circle = R*np.sin(phi)

    for i in range(0, len(t_interp)):
        x = z_interp[i, 0]
        y = z_interp[i, 1]
        theta = z_interp[i, 2]

        x_robot = x + x_circle
        y_robot = y + y_circle

        x2 = x + R*np.cos(theta)
        y2 = y + R*np.sin(theta)

        # 로봇 방향을 나타내는 막대
        line, = plt.plot([x, x2], [y, y2], color='black')
        robot,  = plt.plot(x_robot, y_robot, color='black')

        # 로봇이 그리는 경로
        shape, = plt.plot(z_interp[0:i, 0], z_interp[0:i, 1], color='red',
                          marker='o', markersize=0.5)

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal')
        plt.pause(params.pause)

        line.remove()
        robot.remove()
        shape.remove()

    fig3, (ax3, ax4, ax5) = plt.subplots(nrows=3, ncols=1)

    # e, v, omega = plot_items
    t = np.arange(0, params.t_end, 0.01)
    ax3.set_title('Error (x_ref - x_p)')
    ax3.plot(t, err[:, 0], color='green', label='X err')
    ax3.plot(t, err[:, 1], color='orange', label='Y err')
    ax3.legend(loc='upper right')

    ax4.plot(t, z_dot[:, 0], color='blue')
    ax4.set_title('Control Signal - V')

    ax5.plot(t, z_dot[:, 1], color='red')
    ax5.set_title('Control Signal - W')

    plt.show()


def generate_path(params, path_type='astroid', show_path=False):

    t0 = 0
    tend = params.t_end
    t = np.arange(t0, tend, 0.01)

    if path_type == 'circle':
        R = 1.0
        x_ref = R * np.cos(2 * math.pi * t/tend)
        y_ref = R * np.sin(2 * math.pi * t/tend)

    # generate astroid-shape path
    # (note these are coordinates of point P)
    elif path_type == 'astroid':

        x_center = 0
        y_center = 0
        a = 1
        x_ref = x_center + a * np.cos(2 * math.pi * t/tend) ** 3
        y_ref = y_center + a * np.sin(2 * math.pi * t/tend) ** 3

    if show_path is True:
        fig1, ax1 = plt.subplots()
        ax1.plot(x_ref, y_ref)
        ax1.set_title('Object Path')

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal')

    return x_ref, y_ref


def motion_simulation(params, path):

    px, py = params.px, params.py

    x_ref, y_ref = path
    t = np.arange(0, params.t_end, 0.01)

    # initial state
    z0 = np.array([x_ref[0], y_ref[0], np.pi/2])
    z = np.zeros((len(t), 3))
    z[0] = z0

    # plot items - position error & control signals
    e = np.zeros((len(t), 2))
    e[0] = ([0.0, 0.0])

    # store v, w
    z_dot = np.zeros((len(t), 2))
    z_dot[0] = ([0.0, 0.0])

    for i in range(0, len(t)-1):
        # 1. cur robot state in world frame
        _, _, theta = z0

        # 2. x_c, y_c : world frame point after robot movement P
        x_c, y_c, _ = dd_helper.robot_to_world(z0, params)

        # 3. get error
        error = [x_ref[i+1] - x_c, y_ref[i+1] - y_c]
        e[i+1] = error

        # 4. get u = [v, omega] from the errors
        # b = [ 1.0 + params.Kp * error[0], 5.0 + params.Kp * error[1]]
        b = [params.Kp * error[0], params.Kp * error[1]]
        cos = np.cos(theta)
        sin = np.sin(theta)
        Ainv = np.array([
            [cos-(py/px)*sin, sin+(py/px)*cos],
            [-(1/px)*sin,          (1/px)*cos]
        ])

        u = np.matmul(Ainv, np.transpose(b))
        z_dot[i+1] = u

        # 5. now control the car based on u = [v omega] (euler_integration)
        z0 = dd_helper.euler_integration([t[i], t[i+1]], z0, [u[0], u[1]])
        z[i+1] = z0

    return z, z_dot, e


if __name__ == '__main__':

    params = Parameters()
    # 'astroid' or 'circle'
    path = generate_path(params, path_type='astroid', show_path=True)

    try:
        # pre calculate motion states
        z, z_dot, err = motion_simulation(params, path)
    except Exception as e:
        print(e)
    finally:
        # interpolation for animaltion
        t_interp, z_interp = dd_helper.interpolation(params, z)
        # draw motion
        animate(params, t_interp, z_interp, z_dot, err)
        print('Everything done!')
