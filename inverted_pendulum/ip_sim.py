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
from scipy.integrate import odeint

class Param:

    def __init__(self):
        self.g = 9.81
        self.m = 1
        self.M = 5
        self.L = 2
        self.d = 1
        self.b = 1 # pendulum up (b=1)

        self.pause = 0.01
        self.fps = 10


def dynamics1(m, M, L, d, g):

    A = np.array([
        [0, 1, 0, 0], 
        [0, -1.0*d/M, 1.0*g*m/M, 0], 
        [0, 0, 0, 1], 
        [0, 1.0*d/(L*M), -1.0*g*(M + m)/(L*M), 0]
    ])

    B = np.array([0, 1/M, 0, -1/(M*L)]).reshape((4,1))

    return A, B


def dynamics2(m, M, L, d, g):

    A = np.array([
        [0, 1, 0, 0], 
        [0, -1.0*d/M, 1.0*g*m/M, 0], 
        [0, 0, 0, 1], 
        [0, -1.0*d/(L*M), 1.0*g*(M + m)/(L*M), 0]
    ])

    B = np.array([0, 1/M, 0, 1/(M*L)]).reshape((4,1))

    return A, B


def pendcart(z, t, m, M, L, g, d, u):
    
    x, x_dot, theta, theta_dot = z
    
    dx = z[1]
    ax = 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + g*m*np.sin(2*theta)/2 + u)/(M + m*np.sin(theta)**2)
    omega = z[3]
    alpha = -(1.0*g*(M + m)*np.sin(theta) + 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + u)*np.cos(theta))/(L*(M + m*np.sin(theta)**2))
    
    return dx, ax, omega, alpha

def pendcart_lin(z, t, m, M, L, g, d, u):
    
    z = z.reshape((4,1))
    u = np.zeros((1,1))
    A, B = dynamics2(m, M, L, d, g)

    result = A @ z + B @ u

    return result.reshape((4,)).tolist()


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
        
        plt.pause(params.pause)
        stick.remove()
        ball.remove()
        body.remove()
        
    plt.close()

if __name__ == '__main__':

    params = Param()
    m, M, L, g, d = params.m, params.M, params.L, params.g, params.d
    
    ## Simulate closed-loop system
    t0, tend, N = 0, 10, 100
    tspan = np.linspace(t0, tend, N)    
    
    # Initial condition (x, x_dot, theta, theta_dot)
    z0 = np.array([-1, 0, np.pi+0.1, 0])
    u = 0

    m, M, L, g, d = params.m, params.M, params.L, params.g, params.d
    x = odeint(pendcart, z0, tspan, args=(m, M, L, g, d, u))
    # x = odeint(pendcart_lin, z0, tspan, args=(m, M, L, g, d, u))
    
    animate(tspan, x, params)
