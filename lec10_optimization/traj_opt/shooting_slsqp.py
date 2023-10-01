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

from scipy.integrate import odeint
import scipy.optimize as opt


class Parameter():

    def __init__(self):
        self.D = 5

        # self.N = 5
        # try bigger N!
        self.N = 10

        # self.N = 20
        self.z0 = np.array([0, 0])


def cost(x):
    return x[0]


def car(z, t, t1, t2, u1, u2):

    if (t <= t1 or t > t2):
        u = 0
    else:
        u = u1 + (u2-u1)/(t2-t1)*(t-t1)

    return z[1], u


def simulator(x, z0, N):

    T = x[0]

    # seperate x vals into t, u
    t_sim = np.linspace(0, T, N+1)
    u_sim = x[1:]

    print(f'u_sim: {u_sim}')

    # col : N+1, row : 2
    z_sim = np.zeros((N+1, 2))

    # [0, 0]
    z_sim[0, :] = z0

    for i in range(N):
        t = np.array([t_sim[i], t_sim[i+1]])
        args = (t[0], t[1], u_sim[i], u_sim[i+1])
        # res: input으로 시간 2개, state 2개가 들어가므로
        # output은 2 x 2 행렬이다.
        res = odeint(
            car, z_sim[i], t, args=args, rtol=1e-12, atol=1e-12
        )
        z_sim[i+1] = res[1]

    return z_sim, t_sim, u_sim


def nonlinear_fun(x):

    param = Parameter()
    D, z0, N = param.D, param.z0, param.N

    z_sim, _, _ = simulator(x, z0, N)

    x_end = z_sim[-1, 0] - D
    v_end = z_sim[-1, 1]

    return x_end, v_end


def plot(zz, tt, uu):

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

    # plt.show()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


if __name__ == '__main__':

    param = Parameter()
    D, N, z0 = param.D, param.N, param.z0

    T_opt = 2
    T_min, T_max = 1, 5
    u_min, u_max = -5, 5

    # x0 : [ T, N+1개 u ]
    x0 = np.zeros(N+2)
    x_min = [u_min] * (N+2)
    x_max = [u_max] * (N+2)
    x0[0], x_min[0], x_max[0] = T_opt, T_min, T_max

    limits = opt.Bounds(x_min, x_max)

    constraint = {
        'type': 'eq',
        'fun': nonlinear_fun
    }

    result = opt.minimize(
        cost, x0, method='SLSQP',
        constraints=[constraint],
        options={'ftol': 1e-6, 'disp': True, 'maxiter': 500},
        bounds=limits
    )

    x = result.x
    z_sim, t_sim, u_sim = simulator(x, z0, N)

    plot(z_sim, t_sim, u_sim)
