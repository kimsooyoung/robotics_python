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


def rotation(phi, theta, psi):
    R_x = np.array([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])

    R_y = np.array(
        [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
    )

    R_z = np.array([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])

    return R_z @ R_y @ R_x


def animate(fig_no, phi, theta, psi):
    lx = 0.5
    ly = 0.25
    lz = 0.1
    ll = 1
    lmax = np.max(np.array([lx, ly, lz, ll]))

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
    [m, n] = np.shape(v1)
    R = rotation(phi, theta, psi)

    for i in range(0, m):
        vec = np.array([v0[i, 0], v0[i, 1], v0[i, 2]])
        vec = R.dot(vec)
        v1[i] = vec

    fig = plt.figure(1)
    ax = fig.add_subplot(2, 2, fig_no, projection='3d')

    pc1 = art3d.Poly3DCollection(v1[f], facecolors='blue', alpha=0.25)

    # ax.add_collection(pc0)
    ax.add_collection(pc1)

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

    fac = 180 / np.pi
    phideg = math.trunc(float(phi * fac))
    thetadeg = math.trunc(float(theta * fac))
    psideg = math.trunc(float(psi * fac))
    subtit = (
        'phi=' + str(phideg) + ';' +
        'theta=' + str(thetadeg) + ';' +
        'psi=' + str(psideg) + ';'
    )
    ax.set_title(subtit)
    ax.set_xlim(-lmax, lmax)
    ax.set_ylim(-lmax, lmax)
    ax.set_zlim(-lmax, lmax)
    ax.axis('off')


if __name__ == '__main__':
    phi, theta, psi = 0, 0, 0
    animate(1, phi, theta, psi)

    phi, theta, psi = 0, 0, np.pi / 2
    animate(2, phi, theta, psi)

    phi, theta, psi = 0, np.pi / 2, np.pi / 2
    animate(3, phi, theta, psi)

    phi, theta, psi = np.pi / 2, np.pi / 2, np.pi / 2
    animate(4, phi, theta, psi)

    plt.show()
