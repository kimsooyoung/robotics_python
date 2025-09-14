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
from matplotlib.widgets import Slider
import numpy as np


class Parameters:

    def __init__(self):
        self.a1 = 0
        self.alpha1 = 0
        self.d1 = 0

        self.a2 = 0.2
        self.alpha2 = np.deg2rad(-90)
        self.d2 = 0

        self.a3 = 0.1
        self.alpha3 = np.deg2rad(0)
        self.d3 = 0

        self.a4 = 0.05
        self.alpha4 = 0
        self.d4 = 0

        self.a5 = 0.025
        self.alpha5 = 0.0
        self.d5 = 0

        self.a6 = 0.02
        self.alpha6 = 0.0
        self.d6 = 0

        self.pause = 0.01


def DH2Matrix(a, alpha, d, theta):

    H = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
            [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), np.cos(alpha)*d],
            [0, 0, 0, 1],
        ]
    )
    return H


def plot_with_sliders(params):
    fig = plt.figure(figsize=(12, 10))

    # Create 3D subplot for the manipulator
    ax = fig.add_subplot(111, projection='3d')

    # Create sliders
    ax_theta1 = plt.axes([0.1, 0.9, 0.3, 0.03])
    ax_theta2 = plt.axes([0.1, 0.85, 0.3, 0.03])
    ax_theta3 = plt.axes([0.1, 0.8, 0.3, 0.03])
    ax_theta4 = plt.axes([0.1, 0.75, 0.3, 0.03])
    ax_theta5 = plt.axes([0.1, 0.7, 0.3, 0.03])

    slider_theta1 = Slider(ax_theta1, 'Theta1', -np.pi, np.pi,
                           valinit=0, valfmt='%.2f')
    slider_theta2 = Slider(ax_theta2, 'Theta2', -np.pi, np.pi,
                           valinit=0, valfmt='%.2f')
    slider_theta3 = Slider(ax_theta3, 'Theta3', -np.pi, np.pi,
                           valinit=0, valfmt='%.2f')
    slider_theta4 = Slider(ax_theta4, 'Theta4', -np.pi, np.pi,
                           valinit=0, valfmt='%.2f')
    slider_theta5 = Slider(ax_theta5, 'Theta5', -np.pi, np.pi,
                           valinit=0, valfmt='%.2f')

    # Initialize plot lines
    line1, = ax.plot([], [], [], color='red', linewidth=10)
    line2, = ax.plot([], [], [], color='blue', linewidth=10)
    line3, = ax.plot([], [], [], color='lightblue', linewidth=10)
    line4, = ax.plot([], [], [], color='green', linewidth=10)
    line5, = ax.plot([], [], [], color='yellow', linewidth=10)

    # Set up the plot
    ax.set_xlim([-0.2, 0.5])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-0.2, 0.2])
    ax.view_init(elev=30, azim=-70)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    def update_manipulator(val):
        # Get current slider values
        theta1 = slider_theta1.val
        theta2 = slider_theta2.val
        theta3 = slider_theta3.val
        theta4 = slider_theta4.val
        theta5 = slider_theta5.val
        theta6 = 0  # Keep theta6 fixed at 0

        # Calculate transformation matrices
        H_01 = DH2Matrix(params.a1, params.alpha1, params.d1, theta1)
        H_12 = DH2Matrix(params.a2, params.alpha2, params.d2, theta2)
        H_23 = DH2Matrix(params.a3, params.alpha3, params.d3, theta3)
        H_34 = DH2Matrix(params.a4, params.alpha4, params.d4, theta4)
        H_45 = DH2Matrix(params.a5, params.alpha5, params.d5, theta5)
        H_56 = DH2Matrix(params.a6, params.alpha6, params.d6, theta6)

        # Calculate cumulative transformations
        H_02 = H_01 @ H_12
        H_03 = H_02 @ H_23
        H_04 = H_03 @ H_34
        H_05 = H_04 @ H_45
        H_06 = H_05 @ H_56

        # Extract joint positions
        point1 = H_01[0:3, 3]
        point2 = H_02[0:3, 3]
        point3 = H_03[0:3, 3]
        point4 = H_04[0:3, 3]
        point5 = H_05[0:3, 3]
        point6 = H_06[0:3, 3]

        # Update plot lines
        line1.set_data_3d([point1[0], point2[0]], [point1[1], point2[1]],
                          [point1[2], point2[2]])
        line2.set_data_3d([point2[0], point3[0]], [point2[1], point3[1]],
                          [point2[2], point3[2]])
        line3.set_data_3d([point3[0], point4[0]], [point3[1], point4[1]],
                          [point3[2], point4[2]])
        line4.set_data_3d([point4[0], point5[0]], [point4[1], point5[1]],
                          [point4[2], point5[2]])
        line5.set_data_3d([point5[0], point6[0]], [point5[1], point6[1]],
                          [point5[2], point6[2]])

        fig.canvas.draw_idle()

    # Connect sliders to update function
    slider_theta1.on_changed(update_manipulator)
    slider_theta2.on_changed(update_manipulator)
    slider_theta3.on_changed(update_manipulator)
    slider_theta4.on_changed(update_manipulator)
    slider_theta5.on_changed(update_manipulator)

    # Initial plot
    update_manipulator(None)

    plt.show()


if __name__ == '__main__':
    params = Parameters()

    # Use the interactive plotting function with sliders
    plot_with_sliders(params)
