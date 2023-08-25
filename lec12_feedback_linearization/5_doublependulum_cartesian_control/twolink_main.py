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
from scipy.optimize import fsolve

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

        self.kp1 = 100
        self.kd1 = 2 * np.sqrt(self.kp1)
        self.kp2 = 100
        self.kd2 = 2 * np.sqrt(self.kp2)

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

    G1 = g * (
        c1 * m1 * cos(theta1) + m2 * (c2 * cos(theta1 + theta2) + l * cos(theta1))
    )
    G2 = c2 * g * m2 * cos(theta1 + theta2)

    M = np.array([[M11, M12], [M21, M22]])

    C = np.array([C1, C2])
    G = np.array([G1, G2])

    return M, C, G


def get_tau(
    m1, m2, c1, c2, l, g, I1, I2,
    theta1, theta2, omega1, omega2, kp1, kd1, kp2, kd2,
    q1_p_ref, q1_v_ref, q1_a_ref, q2_p_ref, q2_v_ref, q2_a_ref
):
    qdd_ref = np.array([q1_a_ref, q2_a_ref])
    qd_ref = np.array([q1_v_ref, q2_v_ref])
    q_ref = np.array([q1_p_ref, q2_p_ref])

    q_d = np.array([omega1, omega2])
    q = np.array([theta1, theta2])

    # 여기 주의!
    kp = np.array([[kp1, 0], [0, kp2]])
    kd = np.array([[kd1, 0], [0, kd2]])

    M, C, G = EOM(m1, m2, c1, c2, l, g, I1, I2, theta1, theta2, omega1, omega2)
    tau = M @ (qdd_ref - kp @ (q_d - qd_ref) - kd @ (q - q_ref)) + C + G

    return tau


def twolink_dynamics(z, t, dyn_args, gain_args, ref_args):
    m1, m2, c1, c2, l, g, I1, I2 = dyn_args
    kp1, kd1, kp2, kd2 = gain_args
    q1_p_ref, q1_v_ref, q1_a_ref, q2_p_ref, q2_v_ref, q2_a_ref = ref_args

    theta1, omega1, theta2, omega2 = z

    tau = get_tau(
        m1, m2, c1, c2, l, g, I1, I2,
        theta1, theta2, omega1, omega2, kp1, kd1, kp2, kd2,
        q1_p_ref, q1_v_ref, q1_a_ref, q2_p_ref, q2_v_ref, q2_a_ref
    )
    # noisy tau here
    M, C, G = EOM(m1, m2, c1, c2, l, g, I1, I2, theta1, theta2, omega1, omega2)

    # Ax = b
    A = M
    b = tau - C - G

    x = np.linalg.solve(A, b)

    return [omega1, x[0], omega2, x[1]]


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

    # #plot
    for i in range(0, len(t_interp)):
        theta1 = z_interp[i, 0]
        theta2 = z_interp[i, 2]
        O = np.array([0, 0])
        P = np.array([l * cos(theta1), l * sin(theta1)])
        Q = P + np.array([l * cos(theta1 + theta2), l * sin(theta1 + theta2)])
        G1 = np.array([c1 * cos(theta1), c1 * sin(theta1)])
        G2 = P + np.array([c2 * cos(theta1 + theta2), c2 * sin(theta1 + theta2)])

        (pend1,) = plt.plot([O[0], P[0]], [O[1], P[1]], linewidth=5, color='red')
        (pend2,) = plt.plot([P[0], Q[0]], [P[1], Q[1]], linewidth=5, color='green')
        (com1,) = plt.plot(G1[0], G1[1], color='black', marker='o', markersize=10)
        (com2,) = plt.plot(G2[0], G2[1], color='black', marker='o', markersize=10)
        (endEff,) = plt.plot(Q[0], Q[1], color='black', marker='o', markersize=5)

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal')

        plt.pause(parms.pause)
        pend1.remove()
        pend2.remove()
        com1.remove()
        com2.remove()

    plt.close()


def plot(
    t, z, theta1_ref, theta2_ref, omega1_ref, omega2_ref, T1, T2, tau1_ref, tau2_ref
):
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t, z[:, 0])
    plt.plot(t, theta1_ref, 'r-.')
    plt.ylabel('theta1')
    plt.title('Plot of position vs. time')

    plt.subplot(2, 1, 2)
    plt.plot(t, z[:, 2])
    plt.plot(t, theta2_ref, 'r-.')
    plt.ylabel('theta2')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(t, z[:, 1])
    plt.plot(t, omega1_ref, 'r-.')
    plt.ylabel('theta1dot')
    plt.title('Plot of velocity vs. time')

    plt.subplot(2, 1, 2)
    plt.plot(t, z[:, 3])
    plt.plot(t, omega2_ref, '-.')
    plt.ylabel('theta2dot')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    plt.figure(3)
    plt.subplot(2, 1, 1)
    plt.plot(t, T1)
    plt.plot(t, tau1_ref, 'r-.')
    plt.ylabel('Torque 1')
    plt.title('Plot of acceleration vs. time')

    plt.subplot(2, 1, 2)
    plt.plot(t, T2)
    plt.plot(t, tau2_ref, 'r-.')
    plt.ylabel('Torque 2')
    plt.xlabel('time')

    plt.show(block=False)
    plt.pause(4)
    plt.close()


def get_circle(ts):
    x_ref, y_ref = [], []
    x_d_ref, y_d_ref = [], []
    x_dd_ref, y_dd_ref = [], []

    theta_t = (2 * pi * ts) / max(ts)
    theta_d = 2 * pi / max(ts)

    for theta in theta_t:
        x_ref.append(1 + 0.5 * cos(theta))
        y_ref.append(0.5 + 0.5 * sin(theta))
        x_d_ref.append(-0.5 * sin(theta) * theta_d)
        y_d_ref.append(0.5 * cos(theta) * theta_d)
        x_dd_ref.append(-0.5 * cos(theta) * theta_d**2)
        y_dd_ref.append(-0.5 * sin(theta) * theta_d**2)

    return x_ref, y_ref, x_d_ref, y_d_ref, x_dd_ref, y_dd_ref


def get_eight(ts):
    N = len(ts) - 1
    T = ts[N]
    A = 0.5
    B = A

    a, b = 2, 1
    x0, y0 = 1, 0

    tau = 2 * pi * (-15 * (ts / T) ** 4 + 6 * (ts / T) ** 5 + 10 * (ts / T) ** 3)
    taudot = (
        2
        * pi
        * (
            -15 * 4 * (1 / T) * (ts / T) ** 3
            + 6 * 5 * (1 / T) * (ts / T) ** 4
            + 10 * 3 * (1 / T) * (ts / T) ** 2
        )
    )
    tauddot = (
        2
        * pi
        * (
            -15 * 4 * 3 * (1 / T) ** 2 * (ts / T) ** 2
            + 6 * 5 * 4 * (1 / T) ** 2 * (ts / T) ** 3
            + 10 * 3 * 2 * (1 / T) ** 2 * (ts / T)
        )
    )

    x_ref = x0 + A * sin(a * tau)
    y_ref = y0 + B * cos(b * tau)
    x_d_ref = A * a * cos(a * tau) * taudot
    y_d_ref = -B * b * sin(b * tau) * taudot
    x_dd_ref = -A * a * a * sin(a * tau) * taudot + A * a * cos(a * tau) * tauddot
    y_dd_ref = -B * b * b * sin(b * tau) * taudot - B * b * sin(b * tau) * tauddot

    return x_ref, y_ref, x_d_ref, y_d_ref, x_dd_ref, y_dd_ref


# input : guess theta
# output은 0으로 만들고자 하는 값
def inverse_kinematics(theta, args):
    theta1, theta2 = theta
    l1, l2, x_ref, y_ref = args

    P = np.array([l1 * cos(theta1), l1 * sin(theta1)])
    Q = P + np.array([l2 * cos(theta1 + theta2), l2 * sin(theta1 + theta2)])

    x, y = Q

    return x - x_ref, y - y_ref


def ep_jacobian(l1, l2, theta1, theta2):
    J11 = -l1 * sin(theta1) - l2 * sin(theta1 + theta2)
    J12 = -l2 * sin(theta1 + theta2)
    J21 = l1 * cos(theta1) + l2 * cos(theta1 + theta2)
    J22 = l2 * cos(theta1 + theta2)

    return np.array([J11, J12, J21, J22]).reshape(2, 2)


def ep_jacobian_dot(l1, l2, theta1, theta2, theta1dot, theta2dot):
    Jdot11 = (
        -l1 * theta1dot * cos(theta1)
        - l2 * theta1dot * cos(theta1 + theta2)
        - l2 * theta2dot * cos(theta1 + theta2)
    )
    Jdot12 = -l2 * (theta1dot + theta2dot) * cos(theta1 + theta2)
    Jdot21 = (
        -l1 * theta1dot * sin(theta1)
        - l2 * theta1dot * sin(theta1 + theta2)
        - l2 * theta2dot * sin(theta1 + theta2)
    )
    Jdot22 = -l2 * (theta1dot + theta2dot) * sin(theta1 + theta2)

    return np.array([Jdot11, Jdot12, Jdot21, Jdot22]).reshape(2, 2)


def get_ep_joint_traj(l1, l2, x_ref, y_ref, x_d_ref, y_d_ref, x_dd_ref, y_dd_ref):
    theta1_ref, theta2_ref = [], []
    theta1_d_ref, theta2_d_ref = [], []
    theta1_dd_ref, theta2_dd_ref = [], []

    initial_guess = np.array([0.0, 0.0])

    for i in range(len(x_ref)):
        # q = fslove(IK)
        # q_d = J.inv * x_d
        # q_dd = J.inv * (x_dd - Jdot * q_d)

        theta = fsolve(inverse_kinematics, initial_guess, [l1, l2, x_ref[i], y_ref[i]])
        theta1_ref.append(theta[0])
        theta2_ref.append(theta[1])
        initial_guess = theta

        J = ep_jacobian(l1, l2, theta[0], theta[1])
        x_d = np.array([x_d_ref[i], y_d_ref[i]])
        theta_d = np.linalg.inv(J) @ x_d
        theta1_d_ref.append(theta_d[0])
        theta2_d_ref.append(theta_d[1])

        Jdot = ep_jacobian_dot(l1, l2, theta[0], theta[1], theta_d[0], theta_d[1])
        x_dd = np.array([x_dd_ref[i], y_dd_ref[i]])
        theta_dd = np.linalg.inv(J) @ (x_dd - Jdot @ theta_d)
        theta1_dd_ref.append(theta_dd[0])
        theta2_dd_ref.append(theta_dd[1])

    return (
        theta1_ref,
        theta2_ref,
        theta1_d_ref,
        theta2_d_ref,
        theta1_dd_ref,
        theta2_dd_ref,
    )


if __name__ == '__main__':
    params = Parameters()

    m1, m2, c1, c2, l = params.m1, params.m2, params.c1, params.c2, params.l
    I1, I2, g = params.I1, params.I2, params.g
    kp1, kd1, kp2, kd2 = params.kp1, params.kd1, params.kp2, params.kd2

    N = 200
    t0, t1, tend = 0, 1.5, 3.0
    ts = np.linspace(t0, tend, N)

    # end-point cartesian traj generation
    # x_ref, y_ref, x_d_ref, y_d_ref, x_dd_ref, y_dd_ref = get_circle(ts)
    x_ref, y_ref, x_d_ref, y_d_ref, x_dd_ref, y_dd_ref = get_eight(ts)

    # end-point joint traj generation
    q1_p_ref, q2_p_ref, q1_v_ref, q2_v_ref, q1_a_ref, q2_a_ref = get_ep_joint_traj(
        l, l, x_ref, y_ref, x_d_ref, y_d_ref, x_dd_ref, y_dd_ref
    )

    # initial conditions
    z0 = np.array([q1_p_ref[0], q1_v_ref[0], q2_p_ref[0], q2_v_ref[0]])

    z = np.zeros((len(ts), 4))
    tau = np.zeros((len(ts), 2))
    z[0], tau[0] = z0, np.array([0, 0])

    dyn_args = (m1, m2, c1, c2, l, g, I1, I2)
    gain_args = (kp1, kd1, kp2, kd2)

    for i in range(len(q1_a_ref) - 1):
        ref_args = (
            q1_p_ref[i], q1_v_ref[i], q1_a_ref[i],
            q2_p_ref[i], q2_v_ref[i], q2_a_ref[i],
        )
        args = (dyn_args, gain_args, ref_args)

        t_temp = np.array([ts[i], ts[i + 1]])

        result = odeint(twolink_dynamics, z0, t_temp, args=args)
        tau_temp = get_tau(
            m1, m2, c1, c2, l, g, I1, I2,
            z0[0], z0[2], z0[1], z0[3], kp1, kd1, kp2, kd2,
            q1_p_ref[i], q1_v_ref[i], q1_a_ref[i],
            q2_p_ref[i], q2_v_ref[i], q2_a_ref[i]
        )
        # noisy z here

        z0 = result[-1]
        z[i] = z0
        tau[i] = tau_temp

    animate(ts, z, params)
    plot(
        ts, z,
        q1_p_ref, q2_p_ref, q1_v_ref,
        q2_v_ref, tau[:, 0], tau[:, 1],
        q1_a_ref, q2_a_ref
    )
