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


class Parameters:

    def __init__(self):
        self.a1 = 1
        self.alpha1 = 0.0
        self.d1 = 0
        self.theta1 = 0.0  # user prefered angle

        self.a2 = 1
        self.alpha2 = 0.0
        self.d2 = 0
        self.theta2 = 0.0  # user prefered length

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


def plot(point1, point2):
    # %Draw line from end of link 1 to end of link 2
    plt.plot([0, point1[0]], [0, point1[1]], linewidth=5, color='blue')

    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=5, color='red')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal')

    plt.show()


if __name__ == '__main__':
    params = Parameters()

    a1, alpa1, d1, theta1 = params.a1, params.alpha1, params.d1, 0.0
    a2, alpa2, d2, theta2 = params.a2, params.alpha2, params.d2, 0.0

    # example
    a1, alpa1, d1, theta1 = params.a1, params.alpha1, params.d1, np.pi / 2
    a2, alpa2, d2, theta2 = params.a2, params.alpha2, params.d2, np.pi / 4

    H_01 = DH2Matrix(a1, alpa1, d1, theta1)
    H_12 = DH2Matrix(a2, alpa2, d2, theta2)

    H_01 = H_01
    H_02 = H_01 @ H_12

    point1 = H_01[0:3, 3]
    point2 = H_02[0:3, 3]

    plot(point1, point2)
