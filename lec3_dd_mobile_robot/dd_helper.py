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

import math

import numpy as np
from scipy import interpolate


def euler_integration(tspan, z0, u):

    v = u[0]
    omega = u[1]
    h = tspan[1]-tspan[0]

    x0 = z0[0]
    y0 = z0[1]
    theta0 = z0[2]

    xdot_c = v*math.cos(theta0)
    ydot_c = v*math.sin(theta0)
    thetadot = omega

    x1 = x0 + xdot_c*h
    y1 = y0 + ydot_c*h
    theta1 = theta0 + thetadot*h

    z1 = [x1, y1, theta1]
    return z1


def world_to_robot(state, params):
    x_p, y_p, theta = state
    cos = np.cos(theta)
    sin = np.sin(theta)

    H_robot_world = np.array([
        [cos, -sin, x_p],
        [sin, cos, y_p],
        [0, 0, 1]
    ])

    c = np.array([-params.px, -params.py, 1])

    return np.matmul(H_robot_world, c)


# world frame point after robot frame movement "p"
def robot_to_world(state, params):
    x_c, y_c, theta = state
    cos = np.cos(theta)
    sin = np.sin(theta)

    H_world_robot = np.array([
        [cos, -sin, x_c],
        [sin, cos, y_c],
        [0, 0, 1]
    ])

    p = np.array([params.px, params.py, 1])

    return np.matmul(H_world_robot, p)


def interpolation(params, z):

    # interpolation
    t = np.arange(0, params.t_end, 0.01)

    t_interp = np.arange(0, params.t_end, 1/params.fps)
    f_z1 = interpolate.interp1d(t, z[:, 0])
    f_z2 = interpolate.interp1d(t, z[:, 1])
    f_z3 = interpolate.interp1d(t, z[:, 2])

    z_interp = np.zeros((len(t_interp), 3))
    z_interp[:, 0] = f_z1(t_interp)
    z_interp[:, 1] = f_z2(t_interp)
    z_interp[:, 2] = f_z3(t_interp)

    return t_interp, z_interp
