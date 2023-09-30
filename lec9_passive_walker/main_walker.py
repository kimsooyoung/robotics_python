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

from copy import deepcopy

from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp

from scipy.optimize import fsolve


def cos(x):
    return np.cos(x)


def sin(x):
    return np.sin(x)


class Parameters:

    def __init__(self):
        # m, M : leg mass, body mass
        # I : body moment of inertia
        # g : gravity
        # c, l : leg length, body length
        # gam : slope angle
        # pause, fps : var for animation

        # temporary set small gravity for exercise.
        self.g = 1
        self.m = 0.5
        self.M = 1
        self.I = 0.02
        self.l = 1.0
        self.c = 0.5
        self.gam = 0.1
        self.pause = 0.03
        self.fps = 10


# output이 0이면 충돌이 일어났다는 뜻
def collision(t, z, M, m, I, l, c, g, gam):

    output = 1
    theta1, omega1, theta2, omega2 = z

    # allow legs to pass through for small hip angles
    # (taken care in real walker using stepping stones)
    if (theta1 > -0.05):
        output = 1
    # For lectrue demo case 1.
    # elif (theta2 < 0.05):
    #     output = 1
    else:
        output = 2 * theta1 + theta2

    return output


def single_stance(t, z, M, m, I, l, c, g, gam):

    theta1, omega1, theta2, omega2 = z

    A = np.zeros((2, 2))
    b = np.zeros((2, 1))

    A[0, 0] = 2.0*I + M*l**2 + m*(c - l)**2 + m*(c**2 - 2*c*l*cos(theta2) + l**2)
    A[0, 1] = 1.0*I + c*m*(c - l*cos(theta2))
    A[1, 0] = 1.0*I + c*m*(c - l*cos(theta2))
    A[1, 1] = 1.0*I + c**2*m

    b[0] = -M*g*l*sin(gam - theta1) + c*g*m*sin(gam - theta1) - c*g*m*sin(-gam + theta1 + theta2) - 2*c*l*m*omega1*omega2*sin(theta2) - c*l*m*omega2**2*sin(theta2) - 2*g*l*m*sin(gam - theta1)
    b[1] = 1.0*c*m*(-g*sin(-gam + theta1 + theta2) + l*omega1**2*sin(theta2))

    alpha1, alpha2 = np.linalg.inv(A).dot(b)

    return [omega1, alpha1[0], omega2, alpha2[0]]


def footstrike(t_minus, z_minus, params):

    theta1_n, omega1_n, theta2_n, omega2_n = z_minus

    M = params.M
    m = params.m
    I = params.I
    l = params.l
    c = params.c

    theta1_plus = theta1_n + theta2_n
    theta2_plus = -theta2_n

    J_fs = np.zeros((2, 4))
    A_fs = np.zeros((4, 4))

    b_fs = np.zeros((6, 1))

    J11 = 1
    J12 = 0
    J13 = l*(-cos(theta1_n) + cos(theta1_n + theta2_n))
    J14 = l*cos(theta1_n + theta2_n)
    J21 = 0
    J22 = 1
    J23 = l*(-sin(theta1_n) + sin(theta1_n + theta2_n))
    J24 = l*sin(theta1_n + theta2_n)

    J_fs = np.array([[J11, J12, J13, J14], [J21, J22, J23, J24]])

    A11 = 1.0*M + 2.0*m
    A12 = 0
    A13 = -1.0*M*l*cos(theta1_n) + m*(c - l)*cos(theta1_n) + 1.0*m*(c*cos(theta1_n + theta2_n) - l*cos(theta1_n))
    A14 = 1.0*c*m*cos(theta1_n + theta2_n)
    A21 = 0
    A22 = 1.0*M + 2.0*m
    A23 = -1.0*M*l*sin(theta1_n) + m*(c - l)*sin(theta1_n) + m*(c*sin(theta1_n + theta2_n) - l*sin(theta1_n))
    A24 = 1.0*c*m*sin(theta1_n + theta2_n)
    A31 = -1.0*M*l*cos(theta1_n) + m*(c - l)*cos(theta1_n) + 1.0*m*(c*cos(theta1_n + theta2_n) - l*cos(theta1_n))
    A32 = -1.0*M*l*sin(theta1_n) + m*(c - l)*sin(theta1_n) + m*(c*sin(theta1_n + theta2_n) - l*sin(theta1_n))
    A33 = 2.0*I + M*l**2 + m*(c - l)**2 + m*(c**2 - 2*c*l*cos(theta2_n) + l**2)
    A34 = 1.0*I + c*m*(c - l*cos(theta2_n))
    A41 = 1.0*c*m*cos(theta1_n + theta2_n)
    A42 = 1.0*c*m*sin(theta1_n + theta2_n)
    A43 = 1.0*I + c*m*(c - l*cos(theta2_n))
    A44 = 1.0*I + c**2*m

    A_fs = np.array([
        [A11, A12, A13, A14], [A21, A22, A23, A24], [A31, A32, A33, A34], [A41, A42, A43, A44]
    ])

    M_fs = np.block([
        [A_fs, -np.transpose(J_fs)],
        [J_fs, np.zeros((2, 2))]
    ])

    b_fs = np.block([
        A_fs.dot([0, 0, omega1_n, omega2_n]), 0, 0
    ])

    # x_hs => [vx(+), vy(+), omega1(+), omega2(+) ]
    x_hs = np.linalg.inv(M_fs).dot(b_fs)

    omega1_plus = x_hs[2] + x_hs[3]
    omega2_plus = -x_hs[3]

    return [theta1_plus, omega1_plus, theta2_plus, omega2_plus]


def one_step(z0, t0, params):

    t_start = t0
    t_end = t_start + 4
    t = np.linspace(t_start, t_end, 1001)

    collision.terminal = True
    sol = solve_ivp(
        single_stance, [t_start, t_end], z0, method='RK45', t_eval=t,
        dense_output=True, events=collision, atol=1e-13, rtol=1e-12,
        args=(params.M, params.m, params.I, params.l, params.c, params.g,
              params.gam)
    )

    t = sol.t
    # m : 4 / n : 1001
    m, n = np.shape(sol.y)
    z = np.zeros((n, m))
    z = sol.y.T

    # till single stance
    # foot strike는 z_minus와 t_minus를 준비해서 footstrike 함수에 넣어준다.

    z_minus = np.array(sol.y_events[-1][0, :])
    t_minus = sol.t_events[-1][0]

    z_plus = footstrike(t_minus, z_minus, params)

    t[-1] = t_minus
    z[-1] = z_plus

    return z, t


def n_steps(z0, t0, step_size, params):
    """Walker step main function.

    input
    z0 : initlal state vector [theta1, omega1, theta2, omega2]
    t0 : initial time

    params : parameters params

    output
    z : list of state vector
    t : list of time
    """
    # xh_start, yh_start : hip position
    xh_start, yh_start = 0, params.l * cos(z0[0])

    t = np.array([t0])
    z = np.zeros((1, 6))
    z[0] = np.append(z0, np.array([xh_start, yh_start]))

    for i in range(step_size):
        z_temp, t_temp = one_step(z0, t0, params)

        zz_temp = np.zeros((len(t_temp), 6))

        # append xh, yh - hip position
        for j in range(len(t_temp)):
            xh = xh_start + params.l * sin(z_temp[0, 0]) - params.l * sin(z_temp[j, 0])
            yh = params.l * cos(z_temp[j, 0])
            zz_temp[j, :] = np.append(z_temp[j, :], np.array([xh, yh]))

        z = np.concatenate((z, zz_temp), axis=0)
        t = np.concatenate((t, t_temp), axis=0)

        theta1, omega1, theta2, omega2 = z_temp[-1, 0:4]
        z0 = np.array([theta1, omega1, theta2, omega2])
        t0 = t_temp[-1]

        # one_step에서 zz_temp[-1] 스위칭이 일어나기 때문에 [-2] 사용
        xh_start = zz_temp[-2, 4]

    return z, t


def animate(t, z, parms):
    # interpolation
    data_pts = 1/parms.fps
    t_interp = np.arange(t[0], t[len(t)-1], data_pts)

    [m, n] = np.shape(z)
    z_interp = np.zeros((len(t_interp), n))

    for i in range(0, n):
        f = interpolate.interp1d(t, z[:, i])
        z_interp[:, i] = f(t_interp)

    l = parms.l
    c = parms.c

    min_xh = min(z[:, 4])
    max_xh = max(z[:, 4])

    dist_travelled = max_xh - min_xh
    camera_rate = dist_travelled / len(t_interp)

    window_xmin = -1*l
    window_xmax = 1*l
    window_ymin = -0.1
    window_ymax = 1.1*l

    R1 = np.array([min_xh-l, 0])
    R2 = np.array([max_xh+l, 0])

    # 바닥은 처음에 다 그려버린다.
    ramp, = plt.plot(
        [R1[0], R2[0]],
        [R1[1], R2[1]],
        linewidth=5, color='black'
    )

    # plot body
    for i in range(0, len(t_interp)):
        theta1 = z_interp[i, 0]
        theta2 = z_interp[i, 2]
        xh = z_interp[i, 4]
        yh = z_interp[i, 5]

        H = np.array([xh, yh])
        C1 = np.array([xh+l*sin(theta1), yh-l*cos(theta1)])
        G1 = np.array([xh+c*sin(theta1), yh-c*cos(theta1)])
        C2 = np.array([xh+l*sin(theta1+theta2), yh-l*cos(theta1+theta2)])
        G2 = np.array([xh+c*sin(theta1+theta2), yh-c*cos(theta1+theta2)])

        leg1, = plt.plot([H[0], C1[0]], [H[1], C1[1]], linewidth=5, color='red')
        leg2, = plt.plot([H[0], C2[0]], [H[1], C2[1]], linewidth=5, color='red')
        com1, = plt.plot(G1[0], G1[1], color='black', marker='o', markersize=5)
        com2, = plt.plot(G2[0], G2[1], color='black', marker='o', markersize=5)
        hip, = plt.plot(H[0], H[1], color='black', marker='o', markersize=10)

        # camera_rate 만큼 화면을 오른쪽으로 이동시킨다.
        window_xmin = window_xmin + camera_rate
        window_xmax = window_xmax + camera_rate
        plt.xlim(window_xmin, window_xmax)
        plt.ylim(window_ymin, window_ymax)
        plt.gca().set_aspect('equal')

        plt.pause(parms.pause)
        hip.remove()
        leg1.remove()
        leg2.remove()
        com1.remove()
        com2.remove()

    plt.close()


def fixedpt(z0, params):

    z, t = one_step(z0, 0, params)

    return z[-1, 0] - z0[0], z[-1, 1] - z0[1], \
        z[-1, 2] - z0[2], z[-1, 3] - z0[3]


def partial_jacobian(z, params):

    m = len(z)
    J = np.zeros((m, m))

    epsilon = 1e-5

    for i in range(m):
        # LIST IS IMMUATABLE
        z_minus = deepcopy(z)
        z_plus = deepcopy(z)

        z_minus[i] = z[i] - epsilon
        z_plus[i] = z[i] + epsilon

        z_minus_result, _ = one_step(z_minus, 0, params)
        z_plus_result, _ = one_step(z_plus, 0, params)

        for j in range(m):
            J[i, j] = (z_plus_result[-1, j] - z_minus_result[-1, j]) \
                / (2 * epsilon)

    return J


def plot(t, z):
    plt.figure(1)
    plt.subplot(2, 1, 1)

    plt.plot(t, z[:, 0], 'r--')
    plt.plot(t, z[:, 2], 'b')
    plt.ylabel('theta')

    plt.subplot(2, 1, 2)
    plt.plot(t, z[:, 1], 'r--')
    plt.plot(t, z[:, 3], 'b')
    plt.ylabel('thetadot')
    plt.xlabel('time')

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(t, z[:, 4], 'b')
    plt.ylabel('xh')

    plt.subplot(2, 1, 2)
    plt.plot(t, z[:, 5], 'b')
    plt.ylabel('yh')
    plt.xlabel('time')

    plt.show(block=False)
    plt.pause(3)
    plt.close()


if __name__ == '__main__':

    params = Parameters()

    # # initial state - case 1
    # theta1, omega1, theta2, omega2 = 0.2, -0.25, -0.4, 0.45

    # t0 = 0
    # step_size = 5
    # z0 = np.array([theta1, omega1, theta2, omega2])
    # z, t = n_steps(z0, t0, step_size, params)
    # animate(t, z, params)
    # plot(t, z)

    # initial state - case 2
    theta1, omega1, theta2, omega2 = 0.2, -0.25, -0.4, 0.2

    t0 = 0
    step_size = 10
    z0 = np.array([theta1, omega1, theta2, omega2])

    # 실패하지 않는 초기 조건을 찾아보자.
    z_star = fsolve(fixedpt, z0, params)

    # 해당 초기 조건에 대한 stability를 확인해보자.
    # Jacobian의 determinant를 통해 구해야 하는데,
    # Jacobian을 대수적으로 구할 수 없으므로 수치적으로 구해볼 것이다.
    J_star = partial_jacobian(z_star, params)
    eig_val, eig_vec = np.linalg.eig(J_star)
    print(f'eigVal {eig_val}')
    print(f'eigVec {eig_vec}')
    print(f'max(abs(eigVal)) : {max(np.abs(eig_val))}')

    z, t = n_steps(z_star, t0, step_size, params)
    animate(t, z, params)
    plot(t, z)
