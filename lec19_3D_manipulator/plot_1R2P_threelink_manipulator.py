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
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np


class Parameters:

    def __init__(self):
        self.a1 = 0
        self.alpha1 = 0
        self.d1 = 0.5
        self.theta1 = 0.0  # user prefered angle

        self.a2 = 0
        self.alpha2 = 3 * np.pi / 2
        self.d2 = 0.0  # user prefered length
        self.theta2 = 0.0

        self.a3 = 0
        self.alpha3 = 0.0
        self.d3 = 0.0  # user prefered length
        self.theta3 = 0.0

        self.pause = 0.01


def DH2Matrix(a, alpha, d, theta):
    H_z_theta = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    H_z_d = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])

    H_x_a = np.array([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    H_x_alpha = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0, 1],
        ]
    )

    H = H_z_theta @ H_z_d @ H_x_a @ H_x_alpha

    return H


def plot(point1, point2, point3):
    fig = plt.figure(1)

    # For MacOS Users
    # ax = p3.Axes3D(fig)

    # For Windows/Linux Users
    ax = fig.add_subplot(111, projection='3d')

    (line1,) = ax.plot(
        [0, point1[0]], [0, point1[1]], [0, point1[2]], color='red', linewidth=2
    )
    (line2,) = ax.plot(
        [point1[0], point2[0]],
        [point1[1], point2[1]],
        [point1[2], point2[2]],
        color='blue',
        linewidth=2,
    )
    (line3,) = ax.plot(
        [point2[0], point3[0]],
        [point2[1], point3[1]],
        [point2[2], point3[2]],
        color='lightblue',
        linewidth=2,
    )

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.view_init(elev=30, azim=30)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == '__main__':
    params = Parameters()

    # a1, alpa1, d1, theta1 = params.a1, params.alpha1, params.d1, 0.0
    # a2, alpa2, d2, theta2 = params.a2, params.alpha2, 0.4, params.theta2
    # a3, alpa3, d3, theta3 = params.a3, params.alpha3, 0.25, params.theta3

    # example
    a1, alpa1, d1, theta1 = params.a1, params.alpha1, params.d1, np.pi / 4
    a2, alpa2, d2, theta2 = params.a2, params.alpha2, 0.1, params.theta2
    a3, alpa3, d3, theta3 = params.a3, params.alpha3, 0.6, params.theta3

    H_01 = DH2Matrix(a1, alpa1, d1, theta1)
    H_12 = DH2Matrix(a2, alpa2, d2, theta2)
    H_23 = DH2Matrix(a3, alpa3, d3, theta3)

    H_01 = H_01
    H_02 = H_01 @ H_12
    H_03 = H_02 @ H_23

    point1 = H_01[0:3, 3]
    point2 = H_02[0:3, 3]
    point3 = H_03[0:3, 3]

    plot(point1, point2, point3)
