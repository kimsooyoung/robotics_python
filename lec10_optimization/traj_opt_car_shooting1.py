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

import random

from matplotlib import pyplot as plt
import numpy as np

from scipy.integrate import odeint
import scipy.optimize as opt


class parameters:

    def __init__(self):
        self.D = 5
        self.N = 5

        # z0 = [x0, x0_dot]
        self.z0 = np.array([0, 0])


def cost(x):
    return x[0]


# t는 t1 ~ t2 사이의 값이 된다.
# t1, t2 사이로 u1, u2를 interpolation 시킨 뒤
# 특정 t에 대한 u를 구한다.
# 1차 함수를 만들고 특정 점에 대한 값을 return 하는 것임
# simulator단에서는 u_opt의 값을 최적화해줄 것이다.
# shoothing method이기 때문에 car함수에 EOM이 들어가는 것이 아니라
# interpolation이 들어가는 것임!
def car(z, t, t1, t2, u1, u2):

    xdot = z[1]

    # needed because odeint goes outside t bounds
    if (t > t2 or t <= t1):
        u = 0
    else:
        # f = interpolate.interp1d(t_opt, u_opt)
        # u = f(t)

        # 이게 더 빠르다. (일차함수)
        u = u1 + (u2-u1)/(t2-t1)*(t-t1)

    # z0가 [ x, xdot ]이므로 output은 [ xdot, u ]이다.
    return xdot, u


# [N+1, 2] 크기의 z를 채워넣는 시뮬레이터
# car라는 odeint를 통해 새로운 z를 얻어내고
# 해당 작업을 N번 반복 (z[0]는 초기값이다.)
def simulator(x, z0, N):

    # x = [ T, N+1개 u ]
    T = x[0]

    u_opt = x[1:]
    print(f'u_opt: {u_opt}')

    t_opt = np.linspace(0, T, N+1)

    zz = np.zeros((N+1, 2))
    zz[0, 0], zz[0, 1] = z0[0], z0[1]

    for i in range(0, N):  # goes from 0 to N
        args = (t_opt[i], t_opt[i+1], u_opt[i], u_opt[i+1])
        # res: input으로 시간 2개, state 2개가 들어가므로
        # output은 2*2 행렬이다.
        z = odeint(
            car, zz[i], np.array([t_opt[i], t_opt[i+1]]),
            args, rtol=1e-12, atol=1e-12
        )
        zz[i+1] = z[1]

    return t_opt, zz, u_opt


def nonlinear_fn(x):
    parms = parameters()

    # z0는 시뮬레이터에 들어가는 초기값으로 constraints에서 사용되지 않음
    D, z0, N = parms.D, parms.z0, parms.N

    # zz : [N+1, 2]
    [tt, zz, uu] = simulator(x, z0, N)

    x_end = zz[N, 0] - D
    v_end = zz[N, 1]

    return [x_end, v_end]


def plot(tt, zz, uu):

    plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.plot(tt, zz[:, 0])
    plt.ylabel('x')

    plt.subplot(3, 1, 2)
    plt.plot(tt, zz[:, 1])
    plt.ylabel('xdot')

    plt.subplot(3, 1, 3)
    plt.plot(tt, uu)
    plt.xlabel('t')
    plt.ylabel('u')

    plt.show(block=False)
    plt.pause(10)
    plt.close()


if __name__ == '__main__':

    random.seed(1)

    parms = parameters()
    N = parms.N

    # x0, x_min, x_max 설정
    T_min, T_max = 1, 5
    u_min, u_max = -5, 5

    T_opt = 2
    u_opt = np.zeros(N+1)  # initialize u to zeros

    # 사실 여기 없어도 된다.
    # for i in range(0,N+1):
    #     u_opt[i] = u_min + (u_max-u_min)*random.random()

    # x0 : [ T, N+1 u's ]
    x0 = np.zeros(N+2)
    x_min = np.zeros(N+2)
    x_max = np.zeros(N+2)

    x0[0], x_min[0], x_max[0] = T_opt, T_min, T_max

    for i in range(1, N+2):
        x0[i] = u_opt[i-1]
        x_min[i] = u_min
        x_max[i] = u_max

    limits = opt.Bounds(x_min, x_max)

    eq_cons = {
        'type': 'eq',
        'fun': nonlinear_fn
    }

    res = opt.minimize(
        cost, x0, method='SLSQP',
        constraints=[eq_cons],
        options={'ftol': 1e-6, 'disp': True, 'maxiter': 500},
        bounds=limits
    )

    print(x0)
    x = res.x

    # testing
    z0 = parms.z0
    [tt, zz, uu] = simulator(x, z0, N)
    print(zz[N, 0])
    print(zz[N, 1])

    plot(tt, zz, uu)
