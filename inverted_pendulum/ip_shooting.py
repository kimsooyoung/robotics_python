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

class OptParams:

    def __init__(self):
        self.N = 20
        self.window = 5
        self.dt = 0.1

class Parameters:

    def __init__(self):

        self.g = 9.81
        self.m = 1
        self.M = 5
        self.L = 2
        self.d = 0.1
        self.b = 1 # pendulum up (b=1)
        self.theta_des = np.pi/2

        self.z0 = [-1, 0, np.pi+0.1, 0]
        self.z_end = [1, 0, np.pi, 0]

        self.theta1_mean, self.theta1_dev = 0.0, 0.0
        self.theta1dot_mean, self.theta1dot_dev = 0, 0.5 * 0

        self.pause = 0.01
        self.fps = 10


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

def dynamics_pendcart(z, t, g, m, M, L, d, b, t1, t2, u1, u2):
    
    x, x_dot, theta, theta_dot = z
    u = controller(t, t1, t2, u1, u2)
    
    dx = z[1]
    ax = 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + g*m*np.sin(2*theta)/2 + u)/(M + m*np.sin(theta)**2)
    omega = z[3]
    alpha = -(1.0*g*(M + m)*np.sin(theta) + 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + u)*np.cos(theta))/(L*(M + m*np.sin(theta)**2))
    
    return dx, ax, omega, alpha


def ode_pendcart(z, t, g, m, M, L, d, b, u):
    
    x, x_dot, theta, theta_dot = z
    
    dx = z[1]
    ax = 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + g*m*np.sin(2*theta)/2 + u)/(M + m*np.sin(theta)**2)
    omega = z[3]
    alpha = -(1.0*g*(M + m)*np.sin(theta) + 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + u)*np.cos(theta))/(L*(M + m*np.sin(theta)**2))
    
    return dx, ax, omega, alpha


def simulator(x, z_current):

    t = x[0]
    u_opt = x[1:]

    N = OptParams().N
    dt = OptParams().dt
    parms = Parameters()

    # disturbances
    # theta1_mean, theta1_dev = parms.theta1_mean, parms.theta1_dev
    # theta1dot_mean, theta1dot_dev = parms.theta1dot_dev, parms.theta1dot_dev

    # time setup
    t0, tend = 0, N*dt
    t_opt = np.linspace(t0, tend, N+1)

    # 4 is for theta1 and theta1dot, change according to the system
    z = np.zeros((N+1, 4))
    tau = np.zeros((N+1, 1))
    z0 = z_current
    z[0] = z0

    physical_parms = (parms.g, parms.m, parms.M, parms.L, parms.d, parms.b)

    for i in range(0, N):
        all_parms = physical_parms + (t_opt[i], t_opt[i+1], u_opt[i], u_opt[i+1])

        z_temp = odeint(
            dynamics_pendcart, z0, np.array([t_opt[i], t_opt[i+1]]),
            args=all_parms, atol=1e-13, rtol=1e-13
        )

        t_half = (t_opt[i] + t_opt[i+1])/2
        tau_temp = u_opt[i] + (u_opt[i+1]-u_opt[i])/(t_opt[i+1]-t_opt[i])*(t_half-t_opt[i])

        # z0 = np.array([
        #     z_temp[1, 0] + np.random.normal(theta1_mean, theta1_dev),
        #     z_temp[1, 1] + np.random.normal(theta1dot_mean, theta1dot_dev)
        # ])

        z0 = z_temp[1]
        z[i+1] = z0
        tau[i+1, 0] = tau_temp

    return z[-1], t_opt, z, tau

def pendcart_constraint(x, z0, z_end):

    parms = Parameters()

    z_aft, _, _, _ = simulator(x, z0)

    x_diff = z_aft[0] - z_end[0]
    xdot_diff = z_aft[1] - z_end[1]
    theta_diff = z_aft[2] - z_end[2]
    theta_dot_diff = z_aft[3] - z_end[3]

    return [x_diff, xdot_diff, theta_diff, theta_dot_diff]


def animate(tspan, x, params):
    
    L = params.L
    W = 0.5
    
    plt.xlim(-5, 5)
    plt.ylim(-2.7, 2.7)
    plt.gca().set_aspect('equal')
    
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Inverted Pendulum')
    
    for i in range(len(tspan)):
        stick, = plt.plot(
            [x[i, 0], x[i, 0] + L*np.sin(x[i, 2])], 
            [0, -L*np.cos(x[i, 2])], 
            'b'
        )
        ball, = plt.plot(
            x[i, 0] + L*np.sin(x[i, 2]), 
            -L*np.cos(x[i, 2]), 
            'ro'
        )
        body, = plt.plot(
            [x[i, 0] - W/2, x[i, 0] + W/2],
            [0, 0],
            linewidth=5,
            color='black'
        )

        plt.savefig(f'./images/data{i}.png')

        plt.pause(params.pause)
        stick.remove()
        ball.remove()
        body.remove()
        
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
    dt = opt_params.dt

    time_min, time_max = 1, 20
    u_min, u_max = -20, 120

    # prepare upper/lower bounds
    u_lb = (u_min * np.ones((1, N+1))).flatten()
    u_ub = (u_max * np.ones((1, N+1))).flatten()

    # Initial condition (x, x_dot, theta, theta_dot)
    z0 = parms.z0

    # object state (theta1, theta1dot)
    z_end = parms.z_end

    physical_parms = (parms.g, parms.m, parms.M, parms.L, parms.d, parms.b)

    for i in range(100):

        # temporal control inputs
        u_opt = (u_min + (u_max-u_min) * np.random.rand(1, N+1)).flatten()

        # state x (t, u)
        x0 = [1, *u_opt]
        x_min = [time_min, *u_lb]
        x_max = [time_max, *u_ub]

        limits = opt.Bounds(x_min, x_max)
        constraint = {
            'type': 'eq',
            'fun': pendcart_constraint,
            'args': (z0, z_end),
        }

        # optimize
        result = opt.minimize(
            cost, x0, args=(parms), method='SLSQP',
            constraints=[constraint],
            options={'ftol': 1e-6, 'disp': True, 'maxiter': 10},
            bounds=limits
        )
        opt_state = result.x

        u = opt_state[1]
        all_parms = physical_parms + (u,)

        z_temp = odeint(
            ode_pendcart, z0, np.array([0.0, dt]),
            args=all_parms, atol=1e-13, rtol=1e-13
        )
        z0 = z_temp[1]
        print(z0)

        # print(f'opt_state = {opt_state}')
        # print(f'params.t_opt = {opt_state[0]}')
        # print(f'params.u_opt = {opt_state[1:]}')

    # z_aft, t, z, tau = simulator(opt_state)
    # animate(t, z, parms)
    # plot(t, z, tau, parms)
