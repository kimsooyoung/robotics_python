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

        self.pause = 0.005
        self.fps = 30

        self.Kp_x = 300
        self.Kd_x = 2 * np.sqrt(self.Kp_x)

        self.Kp_y = 300
        self.Kd_y = 2 * np.sqrt(self.Kp_y)

        self.Kp_phi = 2500
        self.Kd_phi = 2 * np.sqrt(self.Kp_phi)


def controller(m, g, l, r, I, gains, refs, x, y, phi, x_d, y_d, phi_d):
    Kp_x, Kp_y, Kp_phi, Kd_x, Kd_y, Kd_phi = gains
    x_ref, y_ref, xd_ref, yd_ref, xdd_ref, ydd_ref = refs

    phi_ref = -(1 / g) * (xdd_ref - Kp_x * (x - x_ref) - Kd_x * (x_d - xd_ref))
    us = m * g + m * (ydd_ref - Kp_y * (y - y_ref) - Kd_y * (y_d - yd_ref))
    ud = -Kp_phi * (phi - phi_ref) - Kd_phi * phi_d

    return us, ud


def bicopter_dynamics(z, t, m, g, l, r, I, gains, refs):
    x, y, phi, x_d, y_d, phi_d = z

    us, ud = controller(m, g, l, r, I, gains, refs, x, y, phi, x_d, y_d, phi_d)

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


def generate_traj8(t, x0, y0):
    # print(len(t))
    T = t[N - 1]
    A = 0.5
    B = A
    a = 2
    b = 1
    pi = np.pi
    tau = 2 * pi * (-15 * (t / T) ** 4 + 6 * (t / T) ** 5 + 10 * (t / T) ** 3)
    taudot = (
        2
        * pi
        * (
            -15 * 4 * (1 / T) * (t / T) ** 3
            + 6 * 5 * (1 / T) * (t / T) ** 4
            + 10 * 3 * (1 / T) * (t / T) ** 2
        )
    )
    tauddot = (
        2
        * pi
        * (
            -15 * 4 * 3 * (1 / T) ** 2 * (t / T) ** 2
            + 6 * 5 * 4 * (1 / T) ** 2 * (t / T) ** 3
            + 10 * 3 * 2 * (1 / T) ** 2 * (t / T)
        )
    )

    x = x0 + A * np.sin(a * tau)
    y = y0 + B * np.cos(b * tau)
    xdot = A * a * np.cos(a * tau) * taudot
    ydot = -B * b * np.sin(b * tau) * taudot
    xddot = -A * a * a * np.sin(a * tau) * taudot + A * a * np.cos(a * tau) * tauddot
    yddot = -B * b * b * np.sin(b * tau) * taudot - B * b * np.sin(b * tau) * tauddot

    if 0:  # code to check the curve
        plt.figure(1)
        plt.plot(x, y)
        plt.ylabel('y')
        plt.xlabel('x')
        plt.title('Plot of trajectory')
        plt.show(block=False)
        plt.pause(2)
        plt.close()

    return x, y, xdot, ydot, xddot, yddot


def generate_traj0(ts, x0, y0):
    r = 0.5

    x_ref, y_ref = [], []
    x_d_ref, y_d_ref = [], []
    x_dd_ref, y_dd_ref = [], []

    theta_t = (2 * np.pi * ts) / max(ts)
    theta_d = 2 * np.pi / max(ts)

    for theta in theta_t:
        x_ref.append(x0 + r * np.cos(theta))
        y_ref.append(y0 + r * np.sin(theta))
        x_d_ref.append(-r * np.sin(theta) * theta_d)
        y_d_ref.append(r * np.cos(theta) * theta_d)
        x_dd_ref.append(-r * np.cos(theta) * theta_d**2)
        y_dd_ref.append(-r * np.sin(theta) * theta_d**2)

    return x_ref, y_ref, x_d_ref, y_d_ref, x_dd_ref, y_dd_ref


def plot(ts, z, x_ref, y_ref, xdot_ref, ydot_ref, tau):
    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(ts, z[:, 0])
    plt.plot(ts, x_ref, 'r-.')
    plt.ylabel('x')
    plt.title('Plot of x vs. time')
    plt.subplot(2, 1, 2)
    plt.plot(ts, z[:, 1])
    plt.plot(ts, y_ref, 'r-.')
    plt.xlabel('time')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(ts, z[:, 3])
    plt.plot(ts, xdot_ref, 'r-.')
    plt.ylabel('xdot')
    plt.title('Plot of velocity (x) vs. time')
    plt.subplot(2, 1, 2)
    plt.plot(ts, z[:, 4])
    plt.plot(ts, ydot_ref, '-.')
    plt.ylabel('ydot')
    plt.xlabel('time')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    plt.figure(3)
    plt.subplot(2, 1, 1)
    plt.plot(ts, tau[:, 0])
    plt.ylabel('Thrust (sum)')
    plt.title('Plot of thrust vs. time')
    plt.subplot(2, 1, 2)
    plt.plot(ts, tau[:, 1])
    plt.ylabel('Thrust (diff)')
    plt.xlabel('time')
    plt.show(block=False)
    plt.pause(4)
    plt.close()


if __name__ == '__main__':
    params = Parameters()
    gains = (
        params.Kp_x,
        params.Kp_y,
        params.Kp_phi,
        params.Kd_x,
        params.Kd_y,
        params.Kd_phi,
    )

    t, tend, N = 0, 5, 100
    ts = np.linspace(t, tend, N)

    # traj gen
    x0, y0 = 0, 0
    x_ref, y_ref, xdot_ref, ydot_ref, xddot_ref, yddot_ref = generate_traj8(ts, x0, y0)
    # x_ref, y_ref, xdot_ref, ydot_ref, xddot_ref, yddot_ref = generate_traj0(ts, x0, y0)

    x, y, phi, x_d, y_d, phi_d = x_ref[0], y_ref[0], 0, 0, 0, 0
    z0 = x, y, phi, x_d, y_d, phi_d
    tau = np.zeros((N, 2))
    z = np.zeros((N, 6))
    z[0] = z0

    for i in range(N - 1):
        refs = x_ref[i], y_ref[i], xdot_ref[i], ydot_ref[i], xddot_ref[i], yddot_ref[i]
        args = params.m, params.g, params.l, params.r, params.I, gains, refs

        t_temp = np.array([ts[i], ts[i + 1]])

        tau_args = args + (*z0,)
        tau_temp = controller(*tau_args)

        result = odeint(bicopter_dynamics, z0, t_temp, args)

        z0 = result[-1]
        z[i + 1] = z0
        tau[i + 1] = tau_temp

    animate(ts, z, params)
    plot(ts, z, x_ref, y_ref, xdot_ref, ydot_ref, tau)
