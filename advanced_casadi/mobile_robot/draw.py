#!/usr/bin/env python
# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib as mpl

class Draw_MPC_point_stabilization_v1(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array, rob_diam=0.3,
                 export_fig=False):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-0.8, 3), ylim=(-0.8, 3.))
        # self.fig.set_dpi(400)
        self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        if export_fig:
            self.ani.save('./v1.gif', writer='imagemagick', fps=100)
        plt.show()

    def animation_init(self):
        # plot target state
        self.target_circle = plt.Circle(self.target_state[:2], self.rob_radius, color='b', fill=False)
        self.ax.add_artist(self.target_circle)
        self.target_arr = mpatches.Arrow(self.target_state[0], self.target_state[1],
                                         self.rob_radius * np.cos(self.target_state[2]),
                                         self.rob_radius * np.sin(self.target_state[2]), width=0.2)
        self.ax.add_patch(self.target_arr)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.target_circle, self.target_arr, self.robot_body, self.robot_arr

    def animation_loop(self, indx):
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]
        self.robot_body.center = position
        # self.ax.add_artist(self.robot_body)
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr, self.robot_body


class Draw_MPC_Obstacle(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array, obstacle: np.array,
                 rob_diam=0.3, export_fig=False):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-0.8, 3), ylim=(-0.8, 3.))
        if obstacle is not None:
            self.obstacle = obstacle
        else:
            print('no obstacle given, break')
        self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        if export_fig:
            self.ani.save('obstacle.gif', writer='imagemagick', fps=100)
        plt.show()

    def animation_init(self):
        # plot target state
        self.target_circle = plt.Circle(self.target_state[:2], self.rob_radius, color='b', fill=False)
        self.ax.add_artist(self.target_circle)
        self.target_arr = mpatches.Arrow(self.target_state[0], self.target_state[1],
                                         self.rob_radius * np.cos(self.target_state[2]),
                                         self.rob_radius * np.sin(self.target_state[2]), width=0.2)
        self.ax.add_patch(self.target_arr)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        self.obstacle_circle = plt.Circle(self.obstacle[:2], self.obstacle[2], color='g', fill=True)
        self.ax.add_artist(self.obstacle_circle)
        return self.target_circle, self.target_arr, self.robot_body, self.robot_arr, self.obstacle_circle

    def animation_loop(self, indx):
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]
        self.robot_body.center = position
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr, self.robot_body


class Draw_MPC_tracking(object):
    def __init__(self, robot_states: list, init_state: np.array, rob_diam=0.3, export_fig=False):
        self.init_state = init_state
        self.robot_states = robot_states
        self.rob_radius = rob_diam
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1.0, 16), ylim=(-0.5, 1.5))
        # self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        if export_fig:
            self.ani.save('tracking.gif', writer='imagemagick', fps=100)
        plt.show()

    def animation_init(self, ):
        # draw target line
        self.target_line = plt.plot([0, 12], [1, 1], '-r')
        # draw the initial position of the robot
        self.init_robot_position = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.init_robot_position)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.target_line, self.init_robot_position, self.robot_body, self.robot_arr

    def animation_loop(self, indx):
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]
        self.robot_body.center = position
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr, self.robot_body


class Draw_FolkLift(object):
    def __init__(self, robot_states: list, initial_state: np.array, export_fig=False):
        self.init_state = initial_state
        self.robot_state_list = robot_states
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1.0, 8.0), ylim=(-0.5, 8.0))

        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_state_list)),
                                                   init_func=self.animation_init, interval=100, repeat=False)
        if export_fig:
            pass
        plt.show()

    def animation_init(self, ):
        x_, y_, angle_ = self.init_state[:3]
        tr = mpl.transforms.Affine2D().rotate_deg_around(x_, y_, angle_)
        t = tr + self.ax.transData
        self.robot_arr = mpatches.Rectangle((x_ - 0.12, y_ - 0.08),
                                             0.24,
                                             0.16,
                                             transform=t,
                                             color='b',
                                             alpha=0.8,
                                             label='DIANA')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr

    def animation_loop(self, indx):
        x_, y_, angle_ = self.robot_state_list[indx][:3]
        angle_ = angle_ * 180 / np.pi
        tr = mpl.transforms.Affine2D().rotate_deg_around(x_, y_, angle_)
        t = tr + self.ax.transData
        self.robot_arr.remove()
        self.robot_arr = mpatches.Rectangle((x_ - 0.12, y_ - 0.08),
                                             0.24,
                                             0.16,
                                             transform=t,
                                             color='b',
                                             alpha=0.8,
                                             label='DIANA')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr