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

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
# import mpl_toolkits.mplot3d.axes3d as p3


class Parameter:

    def __init__(self):
        self.a1 = 0
        self.alpha1 = -np.pi / 2
        self.d1 = 1.3
        self.theta1 = 0.0  # user prefered angle

        self.a2 = 0
        self.alpha2 = np.pi / 2
        self.d2 = 1.4
        self.theta2 = 0.0  # user prefered angle

        self.a3 = 0
        self.alpha3 = 0.0
        self.d3 = 0.0  # user prefered angle
        self.theta3 = -np.pi / 2

        self.a4 = 0
        self.alpha4 = -np.pi / 2
        self.d4 = 0.9
        self.theta4 = 0.0  # user prefered length

        self.a5 = 0
        self.alpha5 = np.pi / 2
        self.d5 = 0
        self.theta5 = 0.0  # user prefered length

        self.a6 = 0
        self.alpha6 = 0
        self.d6 = 0.4
        self.theta6 = 0.0  # user prefered length

        self.pause = 0.05
        self.fps = 30


def DH2Matrix(a, alpha, d, theta):
    H_z_theta = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    H_z_d = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])

    H_x_a = np.array([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    H_x_alpha = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0, 1],
        ]
    )

    H = H_z_theta @ H_z_d @ H_x_a @ H_x_alpha

    return H


def forward_kinematics(initial_guess, X_des, params):
    theta1, theta2, d3, theta4, theta5, theta6 = initial_guess

    a1, alpha1, d1, theta1 = params.a1, params.alpha1, params.d1, theta1
    H_01 = DH2Matrix(a1, alpha1, d1, theta1)

    a2, alpha2, d2, theta2 = params.a2, params.alpha2, params.d2, theta2
    H_12 = DH2Matrix(a2, alpha2, d2, theta2)

    a3, alpha3, d3, theta3 = params.a3, params.alpha3, d3, params.theta3
    H_23 = DH2Matrix(a3, alpha3, d3, theta3)

    a4, alpha4, d4, theta4 = params.a4, params.alpha4, params.d4, theta4
    H_34 = DH2Matrix(a4, alpha4, d4, theta4)

    a5, alpha5, d5, theta5 = params.a5, params.alpha5, params.d5, theta5
    H_45 = DH2Matrix(a5, alpha5, d5, theta5)

    a6, alpha6, d6, theta6 = params.a6, params.alpha6, params.d6, theta6
    H_56 = DH2Matrix(a6, alpha6, d6, theta6)

    H_06 = H_01 @ H_12 @ H_23 @ H_34 @ H_45 @ H_56

    ee_pos = H_06[:3, 3]
    ee_ori = H_06[:3, :3]

    theta = np.arcsin(-ee_ori[2, 0])
    phi = np.arcsin(ee_ori[2, 1] / np.cos(theta))
    psi = np.arcsin(ee_ori[1, 0] / np.cos(theta))

    err = np.array(
        [
            ee_pos[0] - X_des[0],
            ee_pos[1] - X_des[1],
            ee_pos[2] - X_des[2],
            phi - X_des[3],
            theta - X_des[4],
            psi - X_des[5],
        ]
    )

    return err


def animate(X, X_des, order, title, parms):
    theta1, theta2, d3, theta4, theta5, theta6 = X

    a1, alpha1, d1, theta1 = parms.a1, parms.alpha1, parms.d1, theta1
    H01 = DH2Matrix(a1, alpha1, d1, theta1)
    # H^0_1

    a2, alpha2, d2, theta2 = parms.a2, parms.alpha2, parms.d2, theta2
    H12 = DH2Matrix(a2, alpha2, d2, theta2)
    # H^1_2

    a3, alpha3, d3, theta3 = parms.a3, parms.alpha3, d3, parms.theta3
    H23 = DH2Matrix(a3, alpha3, d3, theta3)
    # H^2_3

    a4, alpha4, d4, theta4 = parms.a4, parms.alpha4, parms.d4, theta4
    H34 = DH2Matrix(a4, alpha4, d4, theta4)
    # H^1_2

    a5, alpha5, d5, theta5 = parms.a5, parms.alpha5, parms.d5, theta5
    H45 = DH2Matrix(a5, alpha5, d5, theta5)
    # H^1_2

    a6, alpha6, d6, theta6 = parms.a6, parms.alpha6, parms.d6, theta6
    H56 = DH2Matrix(a6, alpha6, d6, theta6)
    # H^1_2

    # Location of joint 1
    endOfLink1 = H01[0:3, 3]

    # Location of joint 2
    H02 = np.matmul(H01, H12)
    endOfLink2 = H02[0:3, 3]

    # Location of joint 3
    H03 = np.matmul(H02, H23)
    # H01*H12*H23;
    endOfLink3 = H03[0:3, 3]

    # Location of joint 4
    H04 = np.matmul(H03, H34)
    # H01*H12*H23;
    endOfLink4 = H04[0:3, 3]

    # Location of joint 5
    H05 = np.matmul(H04, H45)
    # H01*H12*H23;
    endOfLink5 = H05[0:3, 3]

    # Location of joint 5
    H06 = np.matmul(H05, H56)
    # H01*H12*H23;
    endOfLink6 = H06[0:3, 3]

    # end-effector position and orientation.
    # position_of_end_effector = H06[0:3, 3]
    # orientation_of_end_effector = H06[0:3, 0:3]

    ax = fig.add_subplot(order, projection='3d')

    ax.set_title(title)

    (line1,) = ax.plot(
        [0, endOfLink1[0]],
        [0, endOfLink1[1]],
        [0, endOfLink1[2]],
        color='red',
        linewidth=2,
    )
    (line2,) = ax.plot(
        [endOfLink1[0], endOfLink2[0]],
        [endOfLink1[1], endOfLink2[1]],
        [endOfLink1[2], endOfLink2[2]],
        color='blue',
        linewidth=2,
    )
    (line3,) = ax.plot(
        [endOfLink2[0], endOfLink3[0]],
        [endOfLink2[1], endOfLink3[1]],
        [endOfLink2[2], endOfLink3[2]],
        color='green',
        linewidth=2,
    )
    (line4,) = ax.plot(
        [endOfLink3[0], endOfLink4[0]],
        [endOfLink3[1], endOfLink4[1]],
        [endOfLink3[2], endOfLink4[2]],
        color='yellow',
        linewidth=2,
    )
    (line5,) = ax.plot(
        [endOfLink4[0], endOfLink5[0]],
        [endOfLink4[1], endOfLink5[1]],
        [endOfLink4[2], endOfLink5[2]],
        color=np.array([255 / 255, 165 / 255, 0]),
        linewidth=2,
    )
    (line6,) = ax.plot(
        [endOfLink5[0], endOfLink6[0]],
        [endOfLink5[1], endOfLink6[1]],
        [endOfLink5[2], endOfLink6[2]],
        color='red',
        linewidth=2,
    )

    # point = ax.plot(X_des[0], X_des[1], X_des[2], 'ko')

    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.view_init(azim=63, elev=14)


if __name__ == '__main__':
    params = Parameter()

    # Define the desire end-effector position
    x, y, z = 1.5, 1.5, 2
    phi, theta, psi = 0, 0, 0

    # initial guess for the joint angles
    theta1, theta2, d3 = 0, 0, 0
    theta4, theta5, theta6 = 1, -1, 1

    X_des = np.array([x, y, z, phi, theta, psi])
    initial_guess = np.array([theta1, theta2, d3, theta4, theta5, theta6])

    fig = plt.figure(figsize=(16, 8))
    animate(initial_guess, X_des, 121, 'Initial Guess', params)

    # Solve the inverse kinematics problem
    result = fsolve(
        forward_kinematics, initial_guess, args=(X_des, params), maxfev=500, xtol=1e-6
    )

    animate(result, X_des, 122, 'Final Result', params)
    plt.show()
