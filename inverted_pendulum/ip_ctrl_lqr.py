# Copyright 2023 @RoadBalance
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
import control
from scipy.integrate import odeint

class Param:

    def __init__(self):
        self.g = 9.81
        self.m = 1
        self.M = 5
        self.L = 2
        self.d = 0.1
        self.b = 1 # pendulum up (b=1)

        self.pause = 0.03
        self.fps = 10


def get_control(x, x_ref, K):
    return -K @ (x - x_ref)


def pendcart_linear(z, t, A, B, K=None, z_ref=None):
    
    if isinstance(K, np.ndarray):
        u = get_control(z, z_ref, K).reshape((1,1))
    else:
        u = np.zeros((1,1))
    
    z = z.reshape((4,1))
    
    result = A @ z + B @ u
    return result.reshape((4,)).tolist()

def pendcart_non_linear(z, t, m, M, L, g, d, u_list, K=None, z_ref=None):

    if isinstance(K, np.ndarray):
        u = get_control(z, z_ref, K)[0]
    else:
        u = 0

    u_list.append(u)
    x, x_dot, theta, theta_dot = z
    
    dx = z[1]
    ax = 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + g*m*np.sin(2*theta)/2 + u)/(M + m*np.sin(theta)**2)
    omega = z[3]
    alpha = -(1.0*g*(M + m)*np.sin(theta) + 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + u)*np.cos(theta))/(L*(M + m*np.sin(theta)**2))

    return dx, ax, omega, alpha


# x_dot = 0 / theta = 0 / theta_dot = 0
def dynamics1(m, M, L, d, g):

    A = np.array([
        [0, 1, 0, 0], 
        [0, -1.0*d/M, 1.0*g*m/M, 0], 
        [0, 0, 0, 1], 
        [0, 1.0*d/(L*M), -1.0*g*(M + m)/(L*M), 0]
    ])

    B = np.array([0, 1/M, 0, -1/(M*L)]).reshape((4,1))

    return A, B


# x_dot = 0 / theta = sy.pi / theta_dot = 0
def dynamics2(m, M, L, d, g):

    A = np.array([
        [0, 1, 0, 0], 
        [0, -1.0*d/M, 1.0*g*m/M, 0], 
        [0, 0, 0, 1], 
        [0, -1.0*d/(L*M), 1.0*g*(M + m)/(L*M), 0]
    ])

    B = np.array([0, 1/M, 0, 1/(M*L)]).reshape((4,1))

    return A, B


def animate(tspan, x, params):
    
    L = params.L
    W = 0.5
    
    # plt.xlim(-50, 50)
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
        
        plt.pause(params.pause)
        stick.remove()
        ball.remove()
        body.remove()
        
    plt.close()

def plot(t, z, u):
    # plt.figure(1, figsize=(8, 8))
    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(t, z[:, 0], color='red', label=r'$x$')
    plt.plot(t, z[:, 1], color='blue', label=r'$\dot{x}$')
    plt.plot(t, z[:, 2], color='green', label=r'$\theta$')
    plt.plot(t, z[:, 3], color='orange', label=r'$\dot{\theta}$')
    plt.xlabel('time')
    plt.ylabel('state')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(u, color='brown', label=r'$u$')
    plt.xlabel('time')
    plt.ylabel('state')
    plt.legend()

    plt.show()

if __name__ == '__main__':

    params = Param()
    m, M, L, d, g = params.m, params.M, params.L, params.d, params.g

    # fixed case1. x_dot = 0 / theta = 0 / theta_dot = 0
    A1, B1 = dynamics1(m, M, L, d, g)

    # fixed case2. x_dot = 0 / theta = sy.pi / theta_dot = 0
    A2, B2 = dynamics2(m, M, L, d, g)

    # Define Weighting Matrix
    Q = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0], 
        [0, 0, 10, 0], 
        [0, 0, 0, 100]
    ])
    R = 0.001
    # R = 10

    K1, S1, E1 = control.lqr(A1, B1, Q, R)
    print(f'Gain K = {K1}\n')

    K2, S2, E2 = control.lqr(A2, B2, Q, R)
    eigVal_p, eigVec_p = np.linalg.eig(A2 - B2 @ K2)
    print(f'new eigVal : \n {eigVal_p}')
    print(f'new eigVec : \n {eigVec_p}')
    print(f'Gain K = {K2}\n')

    ## Simulate closed-loop system
    t0, tend, N = 0, 10, 100
    tspan = np.linspace(t0, tend, N)
    z0 = np.array([-1, 0, np.pi+0.1, 0])
    z_ref = np.array([1, 0, np.pi, 0])

    # Simulate linear system
    u_result = []
    z_result = odeint(pendcart_non_linear, z0, tspan, args=(m, M, L, g, d, u_result, K2, z_ref))

    animate(tspan, z_result, params)
    plot(tspan, z_result, u_result)