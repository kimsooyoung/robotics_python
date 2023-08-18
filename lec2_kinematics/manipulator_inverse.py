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
import matrix_helper as mh
import numpy as np
from scipy.optimize import fsolve


class Parameter():

    def __init__(self):
        # define parameters for the two-link
        self.l1 = 1
        self.l2 = 1
        self.O_01 = [0, 0]
        self.O_12 = [self.l1, 0]


def forward_kinematics(l1, l2, theta1, theta2):
    O_01 = [0, 0]
    O_12 = [l1, 0]

    # prepping to get homogenous transformations %%
    H_01 = mh.calc_homogeneous_2d(theta1, O_01)
    H_12 = mh.calc_homogeneous_2d(theta2, O_12)

    # %%%%%%%% origin  in world frame  %%%%%%
    o = [0, 0]

    # %%%%% end of link1 in world frame %%%%
    P1 = np.array([l1, 0, 1])
    P1 = np.transpose(P1)
    P0 = H_01 @ P1
    p = [P0[0], P0[1]]

    # %%%% end of link 2 in world frame  %%%%%%%
    Q2 = np.array([l2, 0, 1])
    Q2 = np.transpose(Q2)
    Q0 = H_01 @ H_12 @ Q2
    q = [Q0[0], Q0[1]]

    return o, p, q


def inverse_kinematics(theta, params):

    theta1, theta2 = theta
    l1, l2, x_ref, y_ref = params

    _, _, q = forward_kinematics(l1, l2, theta1, theta2)

    # return difference btw ref & end-point
    return q[0] - x_ref, q[1] - y_ref


def plot(o, p, q):

    plt.cla()

    # %Draw line from origin to end of link 1
    link1, = plt.plot([o[0], p[0]], [o[1], p[1]], linewidth=5, color='red')
    # %Draw line from end of link 1 to end of link 2
    link2, = plt.plot([p[0], q[0]], [p[1], q[1]], linewidth=5, color='blue')

    # Draw end point
    point, = plt.plot(q[0], q[1], color='black', marker='o', markersize=5)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()
    plt.pause(0.2)
    plt.gca().set_aspect('equal')

    plt.show(block=False)


def main():

    params = Parameter()
    l1, l2 = params.l1, params.l2

    while True:
        print('=== Type New Ref Points ===')
        x_ref = float(input('x_ref : '))
        y_ref = float(input('y_ref : '))

        fsolve_params = [l1, l2, x_ref, y_ref]

        theta = fsolve(inverse_kinematics, [0.01, 0.5], fsolve_params)
        theta1, theta2 = theta
        print(f'theta1 : {theta1}')
        print(f'theta2 : {theta2}')

        o, p, q = forward_kinematics(l1, l2, theta1, theta2)

        plot(o, p, q)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        plt.close()
