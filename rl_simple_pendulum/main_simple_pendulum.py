# Copyright 2025 @RoadBalance
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


class Parameters():

    def __init__(self):
        self.m = 1.0 # mass
        self.l = 0.5 # lengthx
        self.c = 0.0 # coulomb friction coefficient
        self.b = 0.1 # damping friction coefficient
        self.I = self.m * self.l * self.l # inertia
        self.g = 9.81 # gravity
        self.pause = 0.02
        self.fps = 20


### Utils ###
def cos(angle):
    return np.cos(angle)

def sin(angle):
    return np.sin(angle)

def interpolation(t, z, params):

    # interpolation
    t_interp = np.arange(t[0], t[len(t)-1], 1/params.fps)
    [rows, cols] = np.shape(z)
    z_interp = np.zeros((len(t_interp), cols))

    for i in range(0, cols):
        f = interpolate.interp1d(t, z[:, i])
        z_interp[:, i] = f(t_interp)

    return t_interp, z_interp

### Main functions ###
def animate(t_interp, z_interp, params):

    l = params.l

    # plot
    for i in range(0, len(t_interp)):

        theta = z_interp[i, 0]

        O = np.array([0, 0])
        P = np.array([l*sin(theta), -l*cos(theta)])

        # origin
        orgin, = plt.plot(
            O[0], O[1], color='red', marker='s', markersize=10
        )

        # pendulum 
        pend, = plt.plot(
            [O[0], P[0]], [O[1], P[1]], linewidth=2.5, color='red'
        )

        # point mass
        com, = plt.plot(
            P[0], P[1], color='black', marker='o', markersize=15
        )

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.gca().set_aspect('equal')

        plt.pause(params.pause)

        if (i < len(t_interp)-1):
            pend.remove()
            com.remove()

    plt.show(block=False)
    plt.pause(1)
    plt.close()

    # result plotting
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t, z[:, 0], color='red', label=r'$\theta$')
    plt.ylabel('angle')
    plt.legend(loc='upper left')

    plt.subplot(2, 1, 2)
    plt.plot(t, z[:, 1], color='blue', label=r'$\omega$')
    plt.xlabel('t')
    plt.ylabel('angular rate')
    plt.legend(loc='upper left')

    plt.show()

def simple_pendulum(z0, t, m, l, c, b, g):

    theta, omega = z0
    torque = 0

    theta_dd = (torque - m*g*l*sin(theta) - b*omega - np.sign(omega)*c) / (m*l*l)

    return [omega, theta_dd]

if __name__ == '__main__':

    params = Parameters()

    t = np.linspace(0, 10, 500)

    # initlal state
    z0 = np.array([np.pi/4, 0.001])
    all_params = (
        params.m, params.l,
        params.c, params.b,
        params.g
    )
    z = odeint(simple_pendulum, z0, t, args=all_params)

    t_interp, z_interp = interpolation(t, z, params)
    animate(t_interp, z_interp, params)
