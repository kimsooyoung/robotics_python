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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import numpy as np
from scipy import interpolate
from scipy.integrate import odeint


def cos(angle):
    return np.cos(angle)


def sin(angle):
    return np.sin(angle)


def rotation(phi, theta, psi):
    R_x = np.array([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])

    R_y = np.array(
        [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
    )

    R_z = np.array([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])

    return R_z @ R_y @ R_x


class Parameters:

    def __init__(self):
        self.m = 2
        self.lx, self.ly, self.lz = 1, 0.5, 0.25

        self.Ixx = self.m * (self.ly**2 + self.lz**2) / 12
        self.Iyy = self.m * (self.lx**2 + self.lz**2) / 12
        self.Izz = self.m * (self.lx**2 + self.ly**2) / 12

        self.g = 9.81

        self.pause = 0.01
        self.fps = 30


def free_falling_eom(z, t, m, g, lx, ly, lz, Ixx, Iyy, Izz):
    # Ax = b
    x, y, z, phi, theta, psi, vx, vy, vz, phi_d, theta_d, psi_d = z

    A = np.zeros((6, 6))

    A[0, 0] = m
    A[0, 1] = 0
    A[0, 2] = 0
    A[0, 3] = 0
    A[0, 4] = 0
    A[0, 5] = 0
    A[1, 0] = 0
    A[1, 1] = m
    A[1, 2] = 0
    A[1, 3] = 0
    A[1, 4] = 0
    A[1, 5] = 0
    A[2, 0] = 0
    A[2, 1] = 0
    A[2, 2] = m
    A[2, 3] = 0
    A[2, 4] = 0
    A[2, 5] = 0
    A[3, 0] = 0
    A[3, 1] = 0
    A[3, 2] = 0
    A[3, 3] = 1.0 * Ixx
    A[3, 4] = 0
    A[3, 5] = -1.0 * Ixx * sin(theta)
    A[4, 0] = 0
    A[4, 1] = 0
    A[4, 2] = 0
    A[4, 3] = 0
    A[4, 4] = 1.0 * Iyy * cos(phi) ** 2 + 1.0 * Izz * sin(phi) ** 2
    A[4, 5] = 0.25 * (Iyy - Izz) * (sin(2 * phi - theta) + sin(2 * phi + theta))
    A[5, 0] = 0
    A[5, 1] = 0
    A[5, 2] = 0
    A[5, 3] = -1.0 * Ixx * sin(theta)
    A[5, 4] = 0.25 * (Iyy - Izz) * (sin(2 * phi - theta) + sin(2 * phi + theta))
    A[5, 5] = (
        1.0 * Ixx * sin(theta) ** 2
        + 1.0 * Iyy * sin(phi) ** 2 * cos(theta) ** 2
        + 1.0 * Izz * cos(phi) ** 2 * cos(theta) ** 2
    )

    b = np.zeros((6, 1))

    b[0] = 0
    b[1] = 0
    b[2] = -g * m
    b[3] = (
        1.0 * Ixx * psi_d * theta_d * cos(theta)
        + Iyy
        * (psi_d * sin(phi) * cos(theta) + theta_d * cos(phi))
        * (psi_d * cos(phi) * cos(theta) - theta_d * sin(phi))
        - 1.0
        * Izz
        * (psi_d * sin(phi) * cos(theta) + theta_d * cos(phi))
        * (psi_d * cos(phi) * cos(theta) - theta_d * sin(phi))
    )
    b[4] = (
        -1.0 * Ixx * phi_d * psi_d * cos(theta)
        + 0.5 * Ixx * psi_d**2 * sin(2 * theta)
        - 0.5 * Iyy * phi_d * psi_d * cos(2 * phi - theta)
        - 0.5 * Iyy * phi_d * psi_d * cos(2 * phi + theta)
        + 1.0 * Iyy * phi_d * theta_d * sin(2 * phi)
        - 0.25 * Iyy * psi_d**2 * sin(2 * theta)
        - 0.125 * Iyy * psi_d**2 * sin(2 * phi - 2 * theta)
        + 0.125 * Iyy * psi_d**2 * sin(2 * phi + 2 * theta)
        + 0.5 * Izz * phi_d * psi_d * cos(2 * phi - theta)
        + 0.5 * Izz * phi_d * psi_d * cos(2 * phi + theta)
        - 1.0 * Izz * phi_d * theta_d * sin(2 * phi)
        - 0.25 * Izz * psi_d**2 * sin(2 * theta)
        + 0.125 * Izz * psi_d**2 * sin(2 * phi - 2 * theta)
        - 0.125 * Izz * psi_d**2 * sin(2 * phi + 2 * theta)
    )
    b[5] = -0.5 * phi_d * (
        Iyy * psi_d * sin(2 * phi - theta)
        + Iyy * psi_d * sin(2 * phi + theta)
        + 2 * Iyy * theta_d * cos(2 * phi)
        - Izz * psi_d * sin(2 * phi - theta)
        - Izz * psi_d * sin(2 * phi + theta)
        - 2 * Izz * theta_d * cos(2 * phi)
    ) * cos(theta) + 0.25 * theta_d * (
        4 * Ixx * phi_d * cos(theta)
        - 4 * Ixx * psi_d * sin(2 * theta)
        + 2 * Iyy * psi_d * sin(2 * theta)
        + Iyy * psi_d * sin(2 * phi - 2 * theta)
        - Iyy * psi_d * sin(2 * phi + 2 * theta)
        + Iyy * theta_d * cos(2 * phi - theta)
        - Iyy * theta_d * cos(2 * phi + theta)
        + 2 * Izz * psi_d * sin(2 * theta)
        - Izz * psi_d * sin(2 * phi - 2 * theta)
        + Izz * psi_d * sin(2 * phi + 2 * theta)
        - Izz * theta_d * cos(2 * phi - theta)
        + Izz * theta_d * cos(2 * phi + theta)
    )

    q_dd = np.linalg.solve(A, b)
    q_dd = np.array(q_dd[:, 0])

    output = np.array(
        [
            vx,
            vy,
            vz,
            phi_d,
            theta_d,
            psi_d,
            q_dd[0],
            q_dd[1],
            q_dd[2],
            q_dd[3],
            q_dd[4],
            q_dd[5],
        ]
    )

    return output


def animate(t, Xpos, Xang, parms):
    # interpolations
    Xpos = np.array(Xpos)  # convert list to ndarray
    Xang = np.array(Xang)
    t_interp = np.arange(t[0], t[len(t) - 1], 1 / parms.fps)
    [m, n] = np.shape(Xpos)
    shape = (len(t_interp), n)
    Xpos_interp = np.zeros(shape)
    Xang_interp = np.zeros(shape)

    for i in range(0, n):
        fpos = interpolate.interp1d(t, Xpos[:, i])
        Xpos_interp[:, i] = fpos(t_interp)
        fang = interpolate.interp1d(t, Xang[:, i])
        Xang_interp[:, i] = fang(t_interp)

    lx, ly, lz = parms.lx, parms.ly, parms.lz
    ll = np.max(np.array([lx, ly, lz])) + 0.1

    # plot
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

    for i in range(0, len(t_interp)):
        x = Xpos_interp[i, 0]
        y = Xpos_interp[i, 1]
        z = Xpos_interp[i, 2]
        phi = Xang_interp[i, 0]
        theta = Xang_interp[i, 1]
        psi = Xang_interp[i, 2]

        v1 = np.zeros(np.shape(v0))
        [m, n] = np.shape(v1)
        R = rotation(phi, theta, psi)
        for i in range(0, m):
            vec = np.array([v0[i, 0], v0[i, 1], v0[i, 2]])
            vec = R.dot(vec)
            v1[i, 0] = vec[0] + x
            v1[i, 1] = vec[1] + y
            v1[i, 2] = vec[2] + z

        fig = plt.figure(1)
        ax = fig.add_subplot(projection='3d')
        pc1 = art3d.Poly3DCollection(
            v1[f], facecolors='blue', alpha=0.25
        )

        ax.add_collection(pc1)

        origin = np.array([0, 0, 0])
        dirn_x = np.array([1, 0, 0])
        dirn_x = R.dot(dirn_x)
        dirn_y = np.array([0, 1, 0])
        dirn_y = R.dot(dirn_y)
        dirn_z = np.array([0, 0, 1])
        dirn_z = R.dot(dirn_z)
        ax.quiver(
            x + origin[0],
            y + origin[1],
            z + origin[2],
            dirn_x[0],
            dirn_x[1],
            dirn_x[2],
            length=ll,
            arrow_length_ratio=0.1,
            normalize=True,
            color='red',
        )
        ax.quiver(
            x + origin[0],
            y + origin[1],
            z + origin[2],
            dirn_y[0],
            dirn_y[1],
            dirn_y[2],
            length=ll,
            arrow_length_ratio=0.1,
            normalize=True,
            color='green',
        )
        ax.quiver(
            x + origin[0],
            y + origin[1],
            z + origin[2],
            dirn_z[0],
            dirn_z[1],
            dirn_z[2],
            length=ll,
            arrow_length_ratio=0.1,
            normalize=True,
            color='blue',
        )

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-3, 8)
        ax.axis('off')

        plt.pause(parms.pause)

    plt.close()


def calc_engery(z, params):
    P_e, K_e, T_e = [], [], []

    m, g, Ixx, Iyy, Izz = params.m, params.g, params.Ixx, params.Iyy, params.Izz

    for state in z:
        x, y, z, phi, theta, psi, vx, vy, vz, phi_d, theta_d, psi_d = state

        T = (
            0.5 * Ixx * (phi_d - psi_d * sin(theta)) ** 2
            + 0.5 * Iyy * (psi_d * sin(phi) * cos(theta) + theta_d * cos(phi)) ** 2
            + 0.5 * Izz * (psi_d * cos(phi) * cos(theta) - theta_d * sin(phi)) ** 2
            + 0.5 * m * (vx**2 + vy**2 + vz**2)
        )

        V = g * m * z

        P_e.append(V)
        K_e.append(T)
        T_e.append(V + T)

    return P_e, K_e, T_e


def calc_ang_vel(z, params):
    w_b = np.zeros((len(z), 3))
    w_w = np.zeros((len(z), 3))

    for i, state in enumerate(z):
        _, _, _, phi, theta, psi, _, _, _, phi_d, theta_d, psi_d = state
        rates = np.array([phi_d, theta_d, psi_d])

        R_we = np.zeros((3, 3))
        R_we[0, 0] = cos(psi) * cos(theta)
        R_we[0, 1] = -sin(psi)
        R_we[0, 2] = 0
        R_we[1, 0] = sin(psi) * cos(theta)
        R_we[1, 1] = cos(psi)
        R_we[1, 2] = 0
        R_we[2, 0] = -sin(theta)
        R_we[2, 1] = 0
        R_we[2, 2] = 1
        w_w[i] = R_we.dot(rates)

        R_be = np.zeros((3, 3))
        R_be[0, 0] = 1
        R_be[0, 1] = 0
        R_be[0, 2] = -sin(theta)
        R_be[1, 0] = 0
        R_be[1, 1] = cos(phi)
        R_be[1, 2] = sin(phi) * cos(theta)
        R_be[2, 0] = 0
        R_be[2, 1] = -sin(phi)
        R_be[2, 2] = cos(phi) * cos(theta)
        w_b[i] = R_be.dot(rates)

    return w_b, w_w


def plot(ts, z, PE, KE, TE, w_b, w_w):
    ax = plt.figure(2, figsize=(15, 7))

    plt.subplot(2, 3, 1)
    plt.plot(ts, z[:, 0])
    plt.plot(ts, z[:, 1])
    plt.plot(ts, z[:, 2])
    plt.ylabel('linear position')

    plt.subplot(2, 3, 4)
    plt.plot(ts, z[:, 6])
    plt.plot(ts, z[:, 7])
    plt.plot(ts, z[:, 8])
    plt.xlabel('time')
    plt.ylabel('linear velocity')

    plt.subplot(2, 3, 2)
    plt.plot(ts, z[:, 3])
    plt.plot(ts, z[:, 4])
    plt.plot(ts, z[:, 5])
    plt.ylabel('angular position')

    plt.subplot(2, 3, 5)
    plt.plot(ts, z[:, 9])
    plt.plot(ts, z[:, 10])
    plt.plot(ts, z[:, 11])
    plt.xlabel('time')
    plt.ylabel('angular velocity')

    plt.subplot(2, 3, 3)
    plt.plot(ts, w_w[:, 0])
    plt.plot(ts, w_w[:, 1])
    plt.plot(ts, w_w[:, 2])
    ax.legend(['x', 'y', 'z'])
    plt.ylabel('omega world')

    plt.subplot(2, 3, 6)
    plt.plot(ts, w_b[:, 0])
    plt.plot(ts, w_b[:, 1])
    plt.plot(ts, w_b[:, 2])
    ax.legend(['x', 'y', 'z'])
    plt.ylabel('omega body')
    plt.xlabel('time')

    ax = plt.figure(3)
    plt.plot(ts, PE, 'b-.')
    plt.plot(ts, KE, 'r:')
    plt.plot(ts, TE, 'k')
    plt.xlabel('time')
    plt.ylabel('energy')
    ax.legend(['PE', 'KE', 'TE'])

    plt.show()


if __name__ == '__main__':
    params = Parameters()

    m, g = params.m, params.g
    lx, ly, lz = params.lx, params.ly, params.lz
    Ixx, Iyy, Izz = params.Ixx, params.Iyy, params.Izz

    t0, tend, N = 0, 2, 100
    ts = np.linspace(t0, tend, N)

    x0, y0, z0 = 0, 0, 0
    phi0, theta0, psi0 = 0, 0, 0

    vx0, vy0, vz0 = 0, 0, 10
    phi_d0, theta_d0, psi_d0 = 3, -4, 5

    z = np.zeros((N, 12))
    z0 = np.array(
        [x0, y0, z0, phi0, theta0, psi0, vx0, vy0, vz0, phi_d0, theta_d0, psi_d0]
    )
    z[0] = z0

    args = (m, g, lx, ly, lz, Ixx, Iyy, Izz)

    for i in range(N - 1):
        t_temp = np.array([ts[i], ts[i + 1]])
        result = odeint(free_falling_eom, z0, t_temp, args)

        z0 = result[-1]
        z[i + 1] = z0

    pose = z[:, 0:6]
    position = pose[:, 0:3]
    orientation = pose[:, 3:6]

    # calc potential and kinetic energy and total energy
    P_e, K_e, T_e = calc_engery(z, params)

    # calc body frame angular velocity & world frame angular velocity
    w_b, w_w = calc_ang_vel(z, params)

    animate(ts, position, orientation, params)
    plot(ts, z, P_e, K_e, T_e, w_b, w_w)
