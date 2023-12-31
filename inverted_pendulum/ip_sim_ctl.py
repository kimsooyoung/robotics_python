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
        self.d = 1
        self.b = 1 # pendulum up (b=1)

        self.pause = 0.01
        self.fps = 10

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
    m, M, L, d, g = params.m, params.M, params.L, params.d, params.g
    
    # fixed case1. x_dot = 0 / theta = 0 / theta_dot = 0
    A1, B1 = dynamics1(m, M, L, d, g)

    # fixed case2. x_dot = 0 / theta = sy.pi / theta_dot = 0
    A2, B2 = dynamics2(m, M, L, d, g)

    # Calculate the eigenvalues of A
    eigvals1, eigvecs1 = np.linalg.eig(A1)
    eigvals2, eigvecs2 = np.linalg.eig(A2)

    print("eigenvalues of A1")
    print(eigvals1)
    print("eigenvalues of A2")
    print(eigvals2)

    # Determine Controllability
    C1_o = control.ctrb(A1, B1)
    C2_o = control.ctrb(A2, B2)

    print(f'C1_o rank: {np.linalg.matrix_rank(C1_o)}')
    print(f'C2_o rank: {np.linalg.matrix_rank(C2_o)}')
