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
import scipy.optimize as opt

# x0, x_min(-120), x_max(120) 설정
# t, z0, z_end, u
# limits
# constraint 
#    => pendulum_constraint 
#       => simulator
#    => mimize simulator output & desire result 
#    => this will be constraint
#    => cost (u_opt * u_opt) 
# simulator
#    => onelink_rhs
#       => controller (heuristic controller)


def cos(angle):
    return np.cos(angle)


def sin(angle):
    return np.sin(angle)


class OptParams:

    def __init__(self):
        self.N = 10


class Parameters:

    def __init__(self):
        self.m1 = 1
        self.g = 10
        self.l1 = 1
        self.I1 = 1/12 * (self.m1 * self.l1**2)

        self.kp1 = 200
        self.kd1 = 2 * np.sqrt(self.kp1)

        self.theta_des = np.pi/2
        self.z_end = [np.pi/2, 0]

        self.theta1_mean, self.theta1_dev = 0.0, 0.0
        self.theta1dot_mean, self.theta1dot_dev = 0, 0.5 * 0

        self.pause = 0.05
        self.fps = 30


def cost(x, args):

    N = OptParams().N

    time = x[0]
    dt = time / N
    u_opt = x[1:]

    tau_sum = sum([x*y for x, y in zip(u_opt, u_opt)]) * dt

    return tau_sum + time


# heuristic controller
def controller(t, t1, t2, u1, u2):

    max_val = np.max([u1, u2])
    min_val = np.min([u1, u2])

    tau = u1 + (u2-u1)/(t2-t1)*(t-t1)

    if tau > max_val:
        tau = max_val
    elif tau < min_val:
        tau = min_val

    return tau


def onelink_rhs(z, t, m1, I1, l1, g, t1, t2, u1, u2):

    theta1 = z[0]
    theta1dot = z[1]

    tau = controller(t, t1, t2, u1, u2)

    theta1ddot = (1 / (I1+(m1*l1*l1/4))) * (tau - 0.5*m1*g*l1*cos(theta1))

    zdot = np.array([theta1dot, theta1ddot])

    return zdot


def simulator(x):

    t = x[0]
    u_opt = x[1:]

    N = OptParams().N
    parms = Parameters()

    # disturbances
    theta1_mean, theta1_dev = parms.theta1_mean, parms.theta1_dev
    theta1dot_mean, theta1dot_dev = parms.theta1dot_dev, parms.theta1dot_dev

    # time setup
    t0, tend = 0, t
    t_opt = np.linspace(t0, tend, N+1)

    # 2 is for theta1 and theta1dot, change according to the system
    z = np.zeros((N+1, 2))
    tau = np.zeros((N+1, 1))
    z0 = [-np.pi/2, 0]
    z[0] = z0

    physical_parms = (parms.m1, parms.I1, parms.l1, parms.g)

    for i in range(0, N):
        all_parms = physical_parms + (t_opt[i], t_opt[i+1], u_opt[i], u_opt[i+1])

        z_temp = odeint(
            onelink_rhs, z0, np.array([t_opt[i], t_opt[i+1]]),
            args=all_parms, atol=1e-13, rtol=1e-13
        )

        t_half = (t_opt[i] + t_opt[i+1])/2
        tau_temp = u_opt[i] + (u_opt[i+1]-u_opt[i])/(t_opt[i+1]-t_opt[i])*(t_half-t_opt[i])

        z0 = np.array([
            z_temp[1, 0] + np.random.normal(theta1_mean, theta1_dev),
            z_temp[1, 1] + np.random.normal(theta1dot_mean, theta1dot_dev)
        ])

        z[i+1] = z0
        tau[i+1, 0] = tau_temp

    return z[-1], t_opt, z, tau


def pendulum_constraint(x):

    parms = Parameters()
    z_end = parms.z_end

    z_aft, _, _, _ = simulator(x)

    theta1_diff = z_aft[0] - z_end[0]
    theta1dot_diff = z_aft[1] - z_end[1]

    return [theta1_diff, theta1dot_diff]


def animate(t, z, parms):

    # interpolation
    t_interp = np.arange(t[0], t[len(t)-1], 1/parms.fps)

    m, n = np.shape(z)
    shape = (len(t_interp), n)
    z_interp = np.zeros(shape)

    for i in range(n):
        f = interpolate.interp1d(t, z[:, i])
        z_interp[:, i] = f(t_interp)

    l1 = parms.l1

    plt.figure(1)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal')

    # plot
    for i in range(len(t_interp)):
        theta1 = z_interp[i, 0]
        O = np.array([0, 0])
        P = np.array([l1*cos(theta1), l1*sin(theta1)])

        pend1, = plt.plot(
            [O[0], P[0]], [O[1], P[1]], linewidth=5, color='red'
        )

        plt.pause(parms.pause)
        pend1.remove()

    plt.close()


def plot(t, z, T, parms):

    plt.figure(2)

    plt.subplot(3, 1, 1)
    plt.plot(t, parms.theta_des * np.ones(len(t)), 'r-.')
    plt.plot(t, z[:, 0])
    plt.ylabel('theta1')
    plt.title('Plot of position, velocity, and Torque vs. time')

    plt.subplot(3, 1, 2)
    plt.plot(t, z[:, 1])
    plt.ylabel('theta1dot')

    plt.subplot(3, 1, 3)
    plt.plot(t, T[:, 0])
    plt.xlabel('t')
    plt.ylabel('Torque')

    plt.show()


if __name__ == '__main__':

    # Parameters
    parms = Parameters()
    opt_params = OptParams()

    # control sampling
    N = opt_params.N

    time_min, time_max = 1, 4
    u_min, u_max = -20, 120

    # initial state (theta1, theta1dot)
    z0 = [-np.pi/2, 0.0]

    # object state (theta1, theta1dot)
    z_end = parms.z_end

    # temporal control inputs
    u_opt = (u_min + (u_max-u_min) * np.random.rand(1, N+1)).flatten()

    # prepare upper/lower bounds
    u_lb = (u_min * np.ones((1, N+1))).flatten()
    u_ub = (u_max * np.ones((1, N+1))).flatten()

    # state x (t, u)
    x0 = [1, *u_opt]
    x_min = [time_min, *u_lb]
    x_max = [time_max, *u_ub]

    limits = opt.Bounds(x_min, x_max)
    constraint = {
        'type': 'eq',
        'fun': pendulum_constraint
    }

    result = opt.minimize(
        cost, x0, args=(parms), method='SLSQP',
        constraints=[constraint],
        options={'ftol': 1e-6, 'disp': True, 'maxiter': 500},
        bounds=limits
    )
    opt_state = result.x

    print(f'opt_state = {opt_state}')
    print(f'params.t_opt = {opt_state[0]}')
    print(f'params.u_opt = {opt_state[5:]}')

    z_aft, t, z, tau = simulator(opt_state)
    animate(t, z, parms)
    plot(t, z, tau, parms)
