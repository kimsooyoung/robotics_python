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

# [x] EOM 변경해보기
# [x] 그림 수정하기
# [x] linearization - 손으로
# [] control law - mpc
# [] control law - 그 외

class Param:

    def __init__(self):
        self.g = -9.81
        self.m = 1
        self.M = 5
        self.L = 2
        self.d = 1
        self.b = 1 # pendulum up (b=1)

        self.pause = 0.01
        self.fps = 10

## ODE RHS Function Definition
def pendcart(x, t, m, M, L, g, d, u):
    
    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m*L*L*(M+m*(1-Cx**2))

    dx = x[1]
    ax = (1/D)*(-(m**2)*(L**2)*g*Cx*Sx + m*(L**2)*(m*L*(x[3]**2)*Sx - d*x[1])) + m*L*L*(1/D)*u
    omega = x[3]
    alpha = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*(x[3]**2)*Sx - d*x[1])) - m*L*Cx*(1/D)*u;
    
    return dx, ax, omega, alpha

# my EOM
def pendcart2(z, t, m, M, L, g, d, u):
    
    x, x_dot, theta, theta_dot = z
    
    dx = z[1]
    ax = 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + g*m*np.sin(2*theta)/2 + u)/(M + m*np.sin(theta)**2)
    omega = z[3]
    alpha = -(1.0*g*(M + m)*np.sin(theta) + 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + u)*np.cos(theta))/(L*(M + m*np.sin(theta)**2))
    
    return dx, ax, omega, alpha

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
    z0 = np.array([-1, 0, np.pi+0.1, 0]) # Initial condition
    # wr = np.array([1,0,np.pi,0])      # Reference position
    # u = lambda x: -K@(x-wr)           # Control law
    u = 0
    
    # lecture EOM => g = -9.81
    # x = odeint(pendcart, z0, tspan, args=(m, M, L, g, d, u))
    
    # my EOM => g = 9.81
    m, M, L, g, d = params.m, params.M, params.L, -params.g, params.d
    x = odeint(pendcart2, z0, tspan, args=(m, M, L, g, d, u))
    
    animate(tspan, x, params)
    