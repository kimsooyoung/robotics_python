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
import scipy.optimize as opt


class parameters:

    def __init__(self):
        # D : distance between start and end
        # N : number of collocation points
        self.D = 5
        self.N = 12


# cost function : Time 즉, x[0]
def cost(x):
    return x[0]


def nonlinear_fn(x):
    parms = parameters()
    D = parms.D
    N = parms.N

    T = x[0]
    t = np.linspace(0, T, N+1)
    dt = t[1] - t[0]

    # x에서 state들 추출
    pos = [0] * (N+1)
    vel = [0] * (N+1)
    u = [0] * (N+1)
    for i in range(0, N+1):
        pos[i] = x[i+1]
        vel[i] = x[i+1 + N+1]
        u[i] = x[i+1 + N+1 + N+1]

    defect_pos = [0] * N
    defect_vel = [0] * N
    for i in range(0, N):
        defect_pos[i] = pos[i+1] - pos[i] - vel[i]*dt
        defect_vel[i] = vel[i+1] - vel[i] - 0.5*(u[i]+u[i+1])*dt

    ceq = [0] * (2*N + 4)

    # boundary conditions
    # pos(0) : 0
    # vel(0) : 0
    # pos(N + 1) : D
    # vel(N + 1) : 0
    ceq[0] = pos[0]
    ceq[1] = vel[0]
    ceq[2] = pos[N] - D
    ceq[3] = vel[N]

    # defect constraints
    # pos(i+1) - pos(i) - vel(i)*dt = 0
    # vel(i+1) - vel(i) - 0.5*(u(i)+u(i+1))*dt = 0
    for i in range(0, N):
        ceq[i+4] = defect_pos[i]
        ceq[i+N+4] = defect_vel[i]

    return ceq


def plot(T, pos, vel, u, N):

    t = np.linspace(0, T, N+1)
    plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.plot(t, pos)
    plt.ylabel('x')

    plt.subplot(3, 1, 2)
    plt.plot(t, vel)
    plt.ylabel('xdot')

    plt.subplot(3, 1, 3)
    plt.plot(t, u)
    plt.ylabel('u')

    plt.xlabel('t')

    plt.show(block=False)
    plt.pause(10)
    plt.close()


if __name__ == '__main__':

    # define parameters
    parms = parameters()
    N = parms.N
    D = parms.D

    # define constraints
    T_min, T_max = 1, 5
    pos_min, pos_max = 0, D
    vel_min, vel_max = -10, 10
    u_min, u_max = -5, 5

    # define initial guess
    # 일단 아무 숫지만 넣은 것이다.
    # 최적 시간 2, 최적 위치 0, 최적 속도 0, 최적 토크 0
    T_opt = 2
    pos_opt = vel_opt = u_opt = [0] * (N+1)

    # x0 : 모든 state의 initial guess
    # x0 = [ T_opt, pos_opt, vel_opt, u_opt ]
    x0 = [0] * (1+N+1+N+1+N+1)
    x_min = deepcopy(x0)
    x_max = deepcopy(x0)

    # 이제 모두 0이었던 x0에 x_min, x_max를 채워넣자
    x0[0], x_min[0], x_max[0] = T_opt, T_min, T_max

    for i in range(1, 1+N+1):
        x0[i] = pos_opt[i-1]
        x_min[i] = pos_min
        x_max[i] = pos_max
    for i in range(1+N+1, 1+N+1+N+1):
        x0[i] = vel_opt[i-1-N-1]
        x_min[i] = vel_min
        x_max[i] = vel_max
    for i in range(1+N+1+N+1, 1+N+1+N+1+N+1):
        x0[i] = u_opt[i-1-N-1-N-1]
        x_min[i] = u_min
        x_max[i] = u_max
    print(x_min)

    limits = opt.Bounds(x_min, x_max)

    eq_cons = {
        'type': 'eq',
        'fun': nonlinear_fn
    }

    res = opt.minimize(
        cost, x0, method='SLSQP',
        constraints=[eq_cons],
        options={'ftol': 1e-12, 'disp': True, 'maxiter': 500},
        bounds=limits
    )

    # analyze result
    print(x0)
    print(res.x)
    print(res.success)
    print(res.message)
    print(res.status)

    x = res.x

    # T, pos, vel, u 빼내기
    T = x[0]
    pos = [0] * (N+1)
    vel = [0] * (N+1)
    u = [0] * (N+1)
    for i in range(0, N+1):
        pos[i] = x[i+1]
        vel[i] = x[i+1 + N+1]
        u[i] = x[i+1 + N+1 + N+1]

    plot(T, pos, vel, u, N)
