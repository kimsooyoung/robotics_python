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
import numpy as np
from scipy import interpolate
from scipy.integrate import odeint


class parameters:

    def __init__(self):
        self.m = 0.468
        self.Ixx = 4.856 * 1e-3
        self.Iyy = 4.856 * 1e-3
        self.Izz = 8.801 * 1e-3
        self.g = 9.81
        self.l = 0.225
        self.K = 2.980 * 1e-6
        self.b = 1.14 * 1e-7
        self.Ax = 0.25 * 0
        self.Ay = 0.25 * 0
        self.Az = 0.25 * 0
        self.pause = 0.01
        self.fps = 30

        omega = 1
        # speed = 1.0 * omega * np.sqrt(1/self.K)
        speed = 1.075 * omega * np.sqrt(1 / self.K)
        dspeed1 = 0.1 * speed
        dspeed2 = 0.1 * speed
        dspeed3 = 0.1 * speed
        dspeed4 = 0.1 * speed

        self.omega1 = speed + dspeed1
        self.omega2 = speed - dspeed2
        self.omega3 = speed + dspeed3
        self.omega4 = speed - dspeed4

        # self.omega1 = speed+dspeed1
        # self.omega2 = speed-dspeed2
        # self.omega3 = speed+dspeed3
        # self.omega4 = speed-dspeed4


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

    R_temp = np.matmul(R_y, R_x)
    R = np.matmul(R_z, R_temp)
    return R


def quadcopter_dynamics(
    z, t, Ixx, Iyy, Izz, m, g, l, K, b, Ax, Ay, Az, omega1, omega2, omega3, omega4
):
    x, y, z, phi, theta, psi, vx, vy, vz, phi_d, theta_d, psi_d = z

    A = np.zeros((6, 6))
    B = np.zeros((6, 1))

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

    B[0] = -Ax * vx + K * (sin(phi) * sin(psi) + sin(theta) * cos(phi) * cos(psi)) * (
        omega1**2 + omega2**2 + omega3**2 + omega4**2
    )
    B[1] = -Ay * vy - K * (sin(phi) * cos(psi) - sin(psi) * sin(theta) * cos(phi)) * (
        omega1**2 + omega2**2 + omega3**2 + omega4**2
    )
    B[2] = (
        -Az * vz
        + K
        * (omega1**2 + omega2**2 + omega3**2 + omega4**2)
        * cos(phi)
        * cos(theta)
        - g * m
    )
    B[3] = (
        Ixx * psi_d * theta_d * cos(theta)
        + Iyy
        * (psi_d * sin(phi) * cos(theta) + theta_d * cos(phi))
        * (psi_d * cos(phi) * cos(theta) - theta_d * sin(phi))
        - 1.0
        * Izz
        * (psi_d * sin(phi) * cos(theta) + theta_d * cos(phi))
        * (psi_d * cos(phi) * cos(theta) - theta_d * sin(phi))
        - K * l * (omega2**2 - omega4**2) / 2
    )
    B[4] = (
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
        - 0.5 * K * l * omega1**2
        + 0.5 * K * l * omega3**2
    )
    B[5] = (
        b * (omega1**2 - omega2**2 + omega3**2 - omega4**2)
        - 0.5
        * phi_d
        * (
            Iyy * psi_d * sin(2 * phi - theta)
            + Iyy * psi_d * sin(2 * phi + theta)
            + 2 * Iyy * theta_d * cos(2 * phi)
            - Izz * psi_d * sin(2 * phi - theta)
            - Izz * psi_d * sin(2 * phi + theta)
            - 2 * Izz * theta_d * cos(2 * phi)
        )
        * cos(theta)
        + 0.25
        * theta_d
        * (
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
    )

    x = np.linalg.solve(A, B)
    ax, ay, az, phi_dd, theta_dd, psi_dd = (
        x[0, 0],
        x[1, 0],
        x[2, 0],
        x[3, 0],
        x[4, 0],
        x[5, 0],
    )

    return np.array(
        [vx, vy, vz, phi_d, theta_d, psi_d, ax, ay, az, phi_dd, theta_dd, psi_dd]
    )


def animate(t, Xpos, Xang, parms):
    # interpolation
    Xpos = np.array(Xpos)  # convert list to ndarray
    Xang = np.array(Xang)
    t_interp = np.arange(t[0], t[len(t) - 1], 1 / parms.fps)
    [m, n] = np.shape(Xpos)
    shape = (len(t_interp), n)
    Xpos_interp = np.zeros(shape)
    Xang_interp = np.zeros(shape)
    l = parms.l

    for i in range(0, n):
        fpos = interpolate.interp1d(t, Xpos[:, i])
        Xpos_interp[:, i] = fpos(t_interp)
        fang = interpolate.interp1d(t, Xang[:, i])
        Xang_interp[:, i] = fang(t_interp)

    axle_x = np.array([[-l / 2, 0, 0], [l / 2, 0, 0]])
    axle_y = np.array([[0, -l / 2, 0], [0, l / 2, 0]])

    [p2, q2] = np.shape(axle_x)

    for ii in range(0, len(t_interp)):
        x = Xpos_interp[ii, 0]
        y = Xpos_interp[ii, 1]
        z = Xpos_interp[ii, 2]
        phi = Xang_interp[ii, 0]
        theta = Xang_interp[ii, 1]
        psi = Xang_interp[ii, 2]
        R = rotation(phi, theta, psi)

        new_axle_x = np.zeros((p2, q2))
        for i in range(0, p2):
            r_body = axle_x[i, :]
            r_world = R.dot(r_body)
            new_axle_x[i, :] = r_world

        new_axle_x = np.array([x, y, z]) + new_axle_x
        # print(new_axle_x)

        new_axle_y = np.zeros((p2, q2))
        for i in range(0, p2):
            r_body = axle_y[i, :]
            r_world = R.dot(r_body)
            new_axle_y[i, :] = r_world

        new_axle_y = np.array([x, y, z]) + new_axle_y
        # print(new_axle_y)

        fig = plt.figure(1)
        # For MacOS Users
        # ax = p3.Axes3D(fig)

        # For Windows/Linux Users
        ax = fig.add_subplot(111, projection='3d')

        (axle1,) = ax.plot(
            new_axle_x[:, 0], new_axle_x[:, 1], new_axle_x[:, 2], 'ro-', linewidth=3
        )
        (axle2,) = ax.plot(
            new_axle_y[:, 0], new_axle_y[:, 1], new_axle_y[:, 2], 'bo-', linewidth=3
        )

        ll = 0.2
        origin = np.array([0, 0, -0.5])
        dirn_x = np.array([1, 0, 0])
        dirn_y = np.array([0, 1, 0])
        dirn_z = np.array([0, 0, 1])
        ax.quiver(
            origin[0],
            0 + origin[1],
            0 + origin[2],
            dirn_x[0],
            dirn_x[1],
            dirn_x[2],
            length=ll,
            arrow_length_ratio=0.25,
            normalize=True,
            color='red',
        )
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            dirn_y[0],
            dirn_y[1],
            dirn_y[2],
            length=ll,
            arrow_length_ratio=0.25,
            normalize=True,
            color='green',
        )
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            dirn_z[0],
            dirn_z[1],
            dirn_z[2],
            length=ll,
            arrow_length_ratio=0.25,
            normalize=True,
            color='blue',
        )

        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        ax.view_init(azim=-72, elev=20)

        # ax.axis('off');

        plt.pause(parms.pause)

    plt.close()


if __name__ == '__main__':
    params = parameters()
    Ixx, Iyy, Izz = params.Ixx, params.Iyy, params.Izz
    m, g, l, K, b = params.m, params.g, params.l, params.K, params.b
    Ax, Ay, Az = params.Ax, params.Ay, params.Az
    omega1, omega2, omega3, omega4 = (
        params.omega1,
        params.omega2,
        params.omega3,
        params.omega4,
    )

    t0, tend, N = 0, 2, 1000
    ts = np.linspace(t0, tend, N)

    x0, y0, z0 = 0, 0, 0
    phi0, theta0, psi0 = 0, 0, 0
    # phi0, theta0, psi0 = 0, 0.2, 0
    vx0, vy0, vz0 = 0, 0, 0
    phi_d0, theta_d0, psi_d0 = 0, 0, 0
    z0 = np.array(
        [x0, y0, z0, phi0, theta0, psi0, vx0, vy0, vz0, phi_d0, theta_d0, psi_d0]
    )

    # 6 dof
    z = np.zeros((N, 12))
    z[0] = z0

    args = (Ixx, Iyy, Izz, m, g, l, K, b, Ax, Ay, Az, omega1, omega2, omega3, omega4)

    for i in range(N - 1):
        t_temp = np.array([ts[i], ts[i + 1]])
        result = odeint(quadcopter_dynamics, z0, t_temp, args)

        z0 = result[-1]
        z[i + 1] = z0

    Xpos = z[:, 0:3]
    Xang = z[:, 3:6]

    animate(ts, Xpos, Xang, params)
