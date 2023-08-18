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


class Parameter():

    def __init__(self):
        # define parameters for the two-link
        self.l1 = 1
        self.l2 = 1
        self.O_01 = [0, 0]
        self.O_12 = [self.l1, 0]


def plot(H_01, H_12, params):

    plt.cla()

    # %%%%%%%% origin  in world frame  %%%%%%
    o = [0, 0]

    # %%%%% end of link1 in world frame %%%%
    P1 = np.array([params.l1, 0, 1])
    P1 = np.transpose(P1)
    P0 = H_01 @ P1
    p = [P0[0], P0[1]]
    #
    # %%%% end of link 2 in world frame  %%%%%%%
    Q2 = np.array([params.l2, 0, 1])
    Q2 = np.transpose(Q2)
    Q0 = H_01 @ H_12 @ Q2
    q = [Q0[0], Q0[1]]

    # Draw line from origin to end of link 1
    link1, = plt.plot([o[0], p[0]], [o[1], p[1]], linewidth=5, color='red')

    # Draw line from end of link 1 to end of link 2
    link2, = plt.plot([p[0], q[0]], [p[1], q[1]], linewidth=5, color='blue')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()
    plt.gca().set_aspect('equal')
    # plt.axis('square')
    plt.pause(0.3)

    plt.show(block=False)


def main():

    params = Parameter()
    O_01, O_12 = params.O_01, params.O_12

    while True:
        print('Type New Thetas')

        theta1 = float(input('theta1 : '))
        theta2 = float(input('theta2 : '))

        # prepping to get homogenous transformations %%
        H_01 = mh.calc_homogeneous_2d(theta1, O_01)
        H_12 = mh.calc_homogeneous_2d(theta2, O_12)

        plot(H_01, H_12, params)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        plt.close()
