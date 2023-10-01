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

import scipy.optimize as opt


class Parameters:

    def __init__(self):
        # D : distance between start and end
        # N : number of collocation points
        self.D = 5
        self.N = 12


def cost(x):
    # minimize T, Time
    return x[0]


def nonlinear_func(x):

    params = Parameters()

    N = params.N
    T = x[0]
    # N points means N+1 steps
    t = np.linspace(0, T, N+1)
    dt = t[1] - t[0]

    pos = np.zeros(N+1)
    vel = np.zeros(N+1)
    u = np.zeros(N+1)

    # seperate x vals into pos, vel, u
    # i: 0 ~ N
    for i in range(N+1):
        # x[1] ~ x[N+1] : pos
        pos[i] = x[i+1]
        # x[N+2] ~ x[2N+2] : vel
        vel[i] = x[i+N+2]
        # x[2N+3] ~ x[3N+4] : u
        u[i] = x[i+2*N+3]

    # prepare dynamics equations
    defect_pos = np.zeros(N)
    defect_vel = np.zeros(N)

    for i in range(N):
        defect_pos[i] = pos[i+1] - pos[i] - dt * vel[i]
        defect_vel[i] = vel[i+1] - vel[i] - dt * 0.5 * (u[i] + u[i+1])

    # pos eq N ea
    # vel eq N ea
    # pos start, max cond
    # vel start, max cond
    #     => total 2N + 4 ea
    ceq = np.zeros(2*N + 4)

    # pos(0) = 0, pos(N) = D
    # vel(0) = 0, vel(N) = 0
    ceq[0] = pos[0]
    ceq[1] = vel[0]
    ceq[2] = pos[N] - params.D
    ceq[3] = vel[N]

    # dynamics eq
    for i in range(N):
        ceq[i+4] = defect_pos[i]
        ceq[i+N+4] = defect_vel[i]

    return ceq


def plot(T_result, pos_result, vel_result, u_result, N):

    t = np.linspace(0, T_result, N+1)
    plt.figure(1)

    plt.subplot(311)
    plt.plot(t, pos_result)
    plt.ylabel('pos')

    plt.subplot(312)
    plt.plot(t, vel_result)
    plt.ylabel('vel')

    plt.subplot(313)
    plt.plot(t, u_result)
    plt.xlabel('time')
    plt.ylabel('u')

    plt.show()
    plt.pause(10)
    plt.close()


if __name__ == '__main__':

    params = Parameters()
    N = params.N

    # T : time
    T_initial = 2

    # x0 : [ T pos vel u ]
    x0 = np.zeros(3*N + 4)
    x_min = np.zeros(3*N + 4)
    x_max = np.zeros(3*N + 4)

    # boundary conditions 선언
    T_min, T_max = 1, 5
    pos_min, pos_max = 0, params.D
    vel_min, vel_max = -10, 10
    u_min, u_max = -5, 5

    x0[0] = T_initial
    x_min[0] = T_min
    x_max[0] = T_max

    # x_min, x_max에 boundary conditions 추가
    for i in range(1, 1 + N+1):
        x_min[i] = pos_min
        x_max[i] = pos_max
    for i in range(1 + N+1, 1 + 2*N+2):
        x_min[i] = vel_min
        x_max[i] = vel_max
    for i in range(1 + 2*N+2, 1 + 3*N+3):
        x_min[i] = u_min
        x_max[i] = u_max

    print(x_min)
    limits = opt.Bounds(x_min, x_max)

    constraints = {
        'type': 'eq',
        'fun': nonlinear_func
    }

    res = opt.minimize(
        cost, x0, method='SLSQP',
        bounds=limits, constraints=constraints,
        options={'ftol': 1e-12, 'disp': True, 'maxiter': 500}
    )

    x_result = res.x

    # x_result[0] => time
    # x_result[1] ~ x_result[N+1] => pos
    # x_result[N+2] ~ x_result[2N+2] => veㅇl
    # x_result[2N+3] ~ x_result[3N+3] => u
    T_result = x_result[0]
    pos_result = x_result[1:1+N+1]
    vel_result = x_result[1+N+1:1+2*N+2]
    u_result = x_result[1+2*N+2:1+3*N+3]

    plot(T_result, pos_result, vel_result, u_result, N)
