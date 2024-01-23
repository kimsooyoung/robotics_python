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
        self.M = 0.5
        self.m = 0.2
        self.L = 0.3
        self.I = 0.006
        self.d = 0.1
        self.b = 1 # pendulum up (b=1)

        self.pause = 0.005
        self.fps = 10

# x_dot = 0 / theta = 0 / theta_dot = 0
def dynamics1(m, M, L, I, d, g):

    A = np.array([
        [0, 1, 0, 0], 
        [0, -1.0*d*(I + L**2*m)/(I*M + I*m + L**2*M*m), 1.0*L**2*g*m**2/(I*M + I*m + L**2*M*m), 0], 
        [0, 0, 0, 1], 
        [0, 1.0*L*d*m/(I*M + I*m + L**2*M*m), -1.0*L*g*m*(M + m)/(I*M + I*m + L**2*M*m), 0]
    ])

    B = np.array([
        [0], 
        [1.0*(I + L**2*m)/(I*M + I*m + L**2*M*m)], 
        [0], 
        [-1.0*L*m/(I*M + I*m + L**2*M*m)]
    ])

    return A, B


# x_dot = 0 / theta = sy.pi / theta_dot = 0
def dynamics2(m, M, L, I, d, g):
    # ref : https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling

    A = np.array([
        [0, 1, 0, 0], 
        [0, -1.0*d*(I + L**2*m)/(I*M + I*m + L**2*M*m), 1.0*L**2*g*m**2/(I*M + I*m + L**2*M*m), 0], 
        [0, 0, 0, 1], 
        [0, -1.0*L*d*m/(I*M + I*m + L**2*M*m), 1.0*L*g*m*(M + m)/(I*M + I*m + L**2*M*m), 0]
    ])

    B = np.array([
        [0], 
        [1.0*(I + L**2*m)/(I*M + I*m + L**2*M*m)], 
        [0], 
        [1.0*L*m/(I*M + I*m + L**2*M*m)]
    ])

    return A, B


def pendcart(z, t, m, M, L, I, g, d, u):
    
    x, x_dot, theta, theta_dot = z
    
    dx = z[1]
    omega = z[3]
    
    # ax = 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + g*m*np.sin(2*theta)/2 + u)/(M + m*np.sin(theta)**2)
    # alpha = -(1.0*g*(M + m)*np.sin(theta) + 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + u)*np.cos(theta))/(L*(M + m*np.sin(theta)**2))
    
    ax =  1.0*(2.0*L*m*theta_dot**2*np.sin(theta) - 2.0*d*x_dot + g*m*np.sin(2*theta)/2 + 2.0*u)/(2*M + m*np.sin(theta)**2 + m)
    alpha =  1.0*(-g*(M + m)*np.sin(theta) - (1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + u)*np.cos(theta))/(L*(2*M + m*np.sin(theta)**2 + m))

    return dx, ax, omega, alpha

def pendcart_lin(z, t, A1, B1, A2, B2):

    theta = z[2]

    # if theta > 2 * np.pi:
    #     theta = theta - 2 * np.pi
    # else:
    #     theta = theta
    # z = np.array([z[0], z[1], theta, z[3]])

    if abs(theta) < 3.1:
        A = A1
        B = B1
    else:
        A = A2
        B = B2

    z = z.reshape((4,1))
    u = np.zeros((1,1))

    result = A @ z + B @ u
    result = result.reshape((4,)).tolist()

    return result


def animate(tspan, x, params):
    
    L = params.L
    W = 0.1
    
    plt.xlim(-2, 2)
    plt.ylim(-0.7, 0.7)
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
    m, M, L, I, g, d = params.m, params.M, params.L, params.I, params.g, params.d
    
    ## Simulate closed-loop system
    t0, tend, N = 0, 2, 100
    tspan = np.linspace(t0, tend, N)
    
    # Initial condition (x, x_dot, theta, theta_dot)
    z0 = np.array([0, 0, np.pi - 0.1, 0])
    u = 0

    m, M, L, I, g, d = params.m, params.M, params.L, params.I, params.g, params.d
    
    x = odeint(pendcart, z0, tspan, args=(m, M, L, I, g, d, u))
    
    # A1, B1 = dynamics1(m, M, L, I, d, g)
    # A2, B2 = dynamics2(m, M, L, I, d, g)
    # x = odeint(pendcart_lin, z0, tspan, args=(A1, B1, A2, B2))

    print(x)
    # animate(tspan, x, params)
