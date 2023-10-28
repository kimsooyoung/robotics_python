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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import numpy as np


def cos(theta):
    return np.cos(theta)


def sin(theta):
    return np.sin(theta)


def rotation(phi, theta, order):
    R_x = np.array([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])

    R_y = np.array(
        [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
    )

    # R_z = np.array([
    #     [cos(psi), -sin(psi), 0],
    #     [sin(psi),  cos(psi), 0],
    #     [0,            0,         1]
    # ])

    if order == 'xy':
        R = R_x @ R_y
    elif order == 'yx':
        R = R_y @ R_x
    else:
        print('error: 3rd argument to rotation should be xy or yx')
        R = 0

    return R


def animate(fig_no, R, phi, theta, order):
    lx, ly, lz, ll = 0.5, 0.25, 0.1, 1
    lmax = np.max(np.array([lx, ly, lz, ll]))

    # each vertex is a row
    v0 = np.array(
        [
            [-lx, -ly, -lz],
            [lx, -ly, -lz],
            [lx, ly, -lz],
            [-lx, ly, -lz],
            [-lx, -ly, lz],
            [lx, -ly, lz],
            [lx, ly, lz],
            [-lx, ly, lz],
        ]
    )

    # 삼각형의 꼭지점을 나타내는 인덱스
    f = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [1, 2, 6],
            [1, 6, 5],
            [0, 5, 4],
            [0, 1, 5],
            [4, 5, 6],
            [6, 7, 4],
            [3, 7, 6],
            [6, 2, 3],
            [0, 4, 7],
            [7, 3, 0],
        ]
    )

    v1 = np.zeros(np.shape(v0))
    m, n = np.shape(v1)

    for i in range(0, m):
        vec = np.array([v0[i, 0], v0[i, 1], v0[i, 2]])
        v1[i] = R.dot(vec)

    fig = plt.figure(1)
    ax = fig.add_subplot(2, 3, fig_no, projection='3d')

    # print(np.shape(v0[f])) => (12, 3, 3)

    # 8 point will become polygons
    # 직육면체를 그리기 위해 8개의 점을 이용하여 12개의 삼각형을 그린다.
    # 6개의 면을 삼각형으로 표현하기 위해서는 12개가 필요하고, 따라서 f가 길이 8인 것임
    pc1 = art3d.Poly3DCollection(v1[f], facecolors='blue', alpha=0.35)

    # add polygons to axes
    # ax.add_collection(pc0)
    ax.add_collection(pc1)

    # draw axes
    origin = np.array([0, 0, 0])
    dirn_x = np.array([1, 0, 0])
    dirn_x = R.dot(dirn_x)
    dirn_y = np.array([0, 1, 0])
    dirn_y = R.dot(dirn_y)
    dirn_z = np.array([0, 0, 1])
    dirn_z = R.dot(dirn_z)

    ax.quiver(
        origin[0], origin[1], origin[2],
        dirn_x[0], dirn_x[1], dirn_x[2],
        length=1, arrow_length_ratio=0.1,
        normalize=True, color='red',
    )
    ax.quiver(
        origin[0], origin[1], origin[2],
        dirn_y[0], dirn_y[1], dirn_y[2],
        length=1, arrow_length_ratio=0.1,
        normalize=True, color='green',
    )
    ax.quiver(
        origin[0], origin[1], origin[2],
        dirn_z[0], dirn_z[1], dirn_z[2],
        length=1, arrow_length_ratio=0.1,
        normalize=True, color='blue',
    )

    # add title
    fac = 180 / np.pi
    phideg = math.trunc(float(phi * fac))
    thetadeg = math.trunc(float(theta * fac))
    # psideg = math.trunc(float(psi*fac))
    subtit = order + ':phi=' + str(phideg) + ';' + 'theta=' + str(thetadeg) + ';'
    ax.set_title(subtit)

    ax.set_xlim(-lmax, lmax)
    ax.set_ylim(-lmax, lmax)
    ax.set_zlim(-lmax, lmax)
    ax.axis('off')


if __name__ == '__main__':
    # x, y, z => phi, theta, psi

    phi, theta = 0, 0
    R = rotation(phi, theta, 'xy')
    animate(1, R, phi, theta, 'xy')

    phi, theta = np.pi / 2, 0
    R = rotation(phi, theta, 'xy')
    animate(2, R, phi, theta, 'xy')

    phi, theta = np.pi / 2, np.pi / 2
    R = rotation(phi, theta, 'xy')
    animate(3, R, phi, theta, 'xy')

    phi, theta = 0, 0
    R = rotation(phi, theta, 'yx')
    animate(4, R, phi, theta, 'yx')

    phi, theta = 0, np.pi / 2
    R = rotation(phi, theta, 'yx')
    animate(5, R, phi, theta, 'yx')

    phi, theta = np.pi / 2, np.pi / 2
    R = rotation(phi, theta, 'yx')
    animate(6, R, phi, theta, 'yx')

    plt.show()
