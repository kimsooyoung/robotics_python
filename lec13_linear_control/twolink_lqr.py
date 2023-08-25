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

import control
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.integrate import odeint

pi = np.pi


def cos(theta):
    return np.cos(theta)


def sin(theta):
    return np.sin(theta)


class Parameters:

    def __init__(self):
        self.m1 = 1
        self.m2 = 1
        self.l = 1
        self.c1 = self.l / 2
        self.c2 = self.l / 2

        self.I1 = self.m1 * self.l**2 / 12
        self.I2 = self.m2 * self.l**2 / 12

        self.g = 9.81

        self.B = np.array([[1, 0], [0, 1]])
        self.Q = np.eye((4))
        self.R = 1e-2 * np.eye((2))

        # control noise and measurement noise
        self.ctrl_noise_m, self.ctrl_noise_dev = 0, 0
        self.pos_meas_noise_m, self.pos_meas_noise_dev = 0, 0
        self.vel_meas_noise_m, self.vel_meas_noise_dev = 0, 0

        # self.ctrl_noise_m, self.ctrl_noise_dev = 0, 0.1
        # self.pos_meas_noise_m, self.pos_meas_noise_dev = 0, 0.01
        # self.vel_meas_noise_m, self.vel_meas_noise_dev = 0, 0.02

        self.pause = 0.01
        self.fps = 30


def EOM(m1, m2, c1, c2, l, g, I1, I2, theta1, theta2, omega1, omega2):
    M11 = (
        1.0 * I1
        + 1.0 * I2
        + c1**2 * m1
        + m2 * (c2**2 + 2 * c2 * l * cos(theta2) + l**2)
    )
    M12 = 1.0 * I2 + c2 * m2 * (c2 + l * cos(theta2))
    M21 = 1.0 * I2 + c2 * m2 * (c2 + l * cos(theta2))
    M22 = 1.0 * I2 + c2**2 * m2

    C1 = -c2 * l * m2 * omega2 * (2.0 * omega1 + 1.0 * omega2) * sin(theta2)
    C2 = c2 * l * m2 * omega1**2 * sin(theta2)

    G1 = -g * (
        c1 * m1 * sin(theta1) + c2 * m2 * sin(theta1 + theta2) + l * m2 * sin(theta1)
    )
    G2 = -c2 * g * m2 * sin(theta1 + theta2)

    M = np.array([[M11, M12], [M21, M22]])

    C = np.array([C1, C2])
    G = np.array([G1, G2])

    return M, C, G


def linearize(z, m1, m2, c1, c2, l, g, I1, I2, B):
    theta1, theta2, omega1, omega2 = z

    M, _, _ = EOM(m1, m2, c1, c2, l, g, I1, I2, theta1, theta2, omega1, omega2)

    dGdq11 = -g * (
        c1 * m1 * cos(theta1) + c2 * m2 * cos(theta1 + theta2) + l * m2 * cos(theta1)
    )
    dGdq12 = -c2 * g * m2 * cos(theta1 + theta2)
    dGdq21 = -c2 * g * m2 * cos(theta1 + theta2)
    dGdq22 = -c2 * g * m2 * cos(theta1 + theta2)
    dGdq = np.array([[dGdq11, dGdq12], [dGdq21, dGdq22]])

    dGdqdot11 = 0
    dGdqdot12 = 0
    dGdqdot21 = 0
    dGdqdot22 = 0
    dGdqd = np.array([[dGdqdot11, dGdqdot12], [dGdqdot21, dGdqdot22]])

    dCdq11 = 0
    dCdq12 = -c2 * l * m2 * omega2 * (2.0 * omega1 + 1.0 * omega2) * cos(theta2)
    dCdq21 = 0
    dCdq22 = 1.0 * c2 * l * m2 * omega1**2 * cos(theta2)
    dCdq = np.array([[dCdq11, dCdq12], [dCdq21, dCdq22]])

    dCdqdot11 = -2.0 * c2 * l * m2 * omega2 * sin(theta2)
    dCdqdot12 = -2.0 * c2 * l * m2 * (omega1 + omega2) * sin(theta2)
    dCdqdot21 = 2 * c2 * l * m2 * omega1 * sin(theta2)
    dCdqdot22 = 0
    dCdqd = np.array([[dCdqdot11, dCdqdot12], [dCdqdot21, dCdqdot22]])

    Minv = np.linalg.inv(M)

    A_lin = np.block(
        [
            [np.zeros((2, 2)), np.identity(2)],
            [-Minv @ (dCdq + dGdq), -Minv @ (dCdqd + dGdqd)],
        ]
    )

    B_lin = np.block([[np.zeros((2, 2))], [Minv @ B]])

    return A_lin, B_lin


def get_tau(x, K):
    return -K @ x


def twolink_dynamics(z, t, dyn_args, control_args, noise_args):
    m1, m2, c1, c2, l, g, I1, I2 = dyn_args
    K, B = control_args
    disturb1, disturb2 = noise_args

    theta1, omega1, theta2, omega2 = z

    M, C, G = EOM(m1, m2, c1, c2, l, g, I1, I2, theta1, theta2, omega1, omega2)
    # noisy tau here
    u = get_tau(z, K)
    disturb = np.array([disturb1, disturb2])
    tau = B @ u + disturb

    # Ax = b
    A = M
    b = (tau - C - G).reshape(2, 1)
    x = np.linalg.inv(A) @ b

    # caution! order of x is different from z
    return [omega1, omega2, x[0, 0], x[1, 0]]


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
    c1 = parms.c1
    c2 = parms.c2

    # plot
    for i in range(0, len(t_interp)):
        theta1 = z_interp[i, 0]
        theta2 = z_interp[i, 1]
        O = np.array([0, 0])
        P = np.array([-l * sin(theta1), l * cos(theta1)])
        Q = P + np.array([-l * sin(theta1 + theta2), l * cos(theta1 + theta2)])
        G1 = np.array([-c1 * sin(theta1), c1 * cos(theta1)])
        G2 = P + np.array([-c2 * sin(theta1 + theta2), c2 * cos(theta1 + theta2)])

        (pend1,) = plt.plot([O[0], P[0]], [O[1], P[1]], linewidth=5, color='red')
        (pend2,) = plt.plot([P[0], Q[0]], [P[1], Q[1]], linewidth=5, color='blue')
        (com1,) = plt.plot(G1[0], G1[1], color='black', marker='o', markersize=10)
        (com2,) = plt.plot(G2[0], G2[1], color='black', marker='o', markersize=10)

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal')

        plt.pause(parms.pause)
        if i < len(t_interp) - 1:
            pend1.remove()
            pend2.remove()
            com1.remove()
            com2.remove()

    # plt.show()
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def plot(t, z, u):
    plt.figure(1, figsize=(6, 8))

    plt.subplot(4, 1, 1)
    plt.plot(t, z[:, 0], color='red', label='theta1')
    plt.plot(t, z[:, 1], color='blue', label='theta2')
    plt.ylabel('angle')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 2)
    plt.plot(t, z[:, 2], color='red', label='omega1')
    plt.plot(t, z[:, 3], color='blue', label='omega2')
    plt.xlabel('t')
    plt.ylabel('angular rate')
    plt.legend(loc='lower left')

    plt.subplot(4, 1, 3)
    plt.plot(t, u[:, 0], color='green')
    plt.xlabel('t')
    plt.ylabel('torque1')

    plt.subplot(4, 1, 4)
    plt.plot(t, u[:, 1], color='green')
    plt.xlabel('t')
    plt.ylabel('torque2')

    plt.show()


if __name__ == '__main__':
    params = Parameters()

    m1, m2, c1, c2, l = params.m1, params.m2, params.c1, params.c2, params.l
    I1, I2, g = params.I1, params.I2, params.g
    B, Q, R = params.B, params.Q, params.R

    ctrl_noise_m, ctrl_noise_dev = params.ctrl_noise_m, params.ctrl_noise_dev
    pos_meas_noise_m, pos_meas_noise_dev = (
        params.pos_meas_noise_m,
        params.pos_meas_noise_dev,
    )
    vel_meas_noise_m, vel_meas_noise_dev = (
        params.vel_meas_noise_m,
        params.vel_meas_noise_dev,
    )

    # 1. linerize
    z = np.array([0, 0, 0, 0])
    A_lin, B_lin = linearize(z, m1, m2, c1, c2, l, g, I1, I2, B)

    # 2. lqr and get K
    K, S, E = control.lqr(A_lin, B_lin, Q, R)
    print(f'K : {K}')
    print(f'E : {E}')

    # 3. prepare simulation
    N = 100
    t0, tend = 0, 5
    ts = np.linspace(t0, tend, N)

    # initial conditions
    # It'll make double pendulum into straight pose
    pi = np.pi
    # z0 = np.array([0, 0, 0, 0])
    z0 = np.array([pi / 4, 0, 0, 0])
    # z0 = np.array([pi/4, -pi/4, 0, 0])

    z = np.zeros((len(ts), 4))
    tau = np.zeros((len(ts), 2))
    z[0] = z0

    dyn_args = (m1, m2, c1, c2, l, g, I1, I2)
    control_args = (K, B)

    for i in range(N - 1):
        disturb1 = np.random.normal(ctrl_noise_m, ctrl_noise_dev)
        disturb2 = np.random.normal(ctrl_noise_m, ctrl_noise_dev)
        noise_args = (disturb1, disturb2)

        # execute simulation
        t_temp = np.array([ts[i], ts[i + 1]])
        args = (dyn_args, control_args, noise_args)
        result = odeint(twolink_dynamics, z0, t_temp, args=args)
        tau[i] = get_tau(z0, K)

        # noisy z here
        meas_noise = np.array(
            [
                np.random.normal(pos_meas_noise_m, pos_meas_noise_dev),
                np.random.normal(vel_meas_noise_m, vel_meas_noise_dev),
                np.random.normal(pos_meas_noise_m, pos_meas_noise_dev),
                np.random.normal(vel_meas_noise_m, vel_meas_noise_dev),
            ]
        )
        z0 = result[-1] + meas_noise
        z[i] = z0

    # animate(ts,z,params)
    plot(ts, z, tau)
