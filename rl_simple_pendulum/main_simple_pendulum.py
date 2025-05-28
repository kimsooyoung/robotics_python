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
import scipy.linalg

from scipy import interpolate
from scipy.integrate import odeint


class Parameters():

    def __init__(self):
        # self.m = 1.0 # mass
        # self.l = 0.5 # length
        # self.c = 0.0 # coulomb friction coefficient
        # self.b = 0.1 # damping friction coefficient
        # self.I = self.m * self.l * self.l # inertia
        # self.g = 9.81 # gravity
        # self.pause = 0.02
        # self.fps = 20

        self.m = 0.3 # mass
        self.l = 1.0 # length
        self.c = 0.0 # coulomb friction coefficient
        self.b = 0.1 # damping friction coefficient
        self.I = self.m * self.l * self.l # inertia
        self.g = 9.81 # gravity
        self.max_torque = 2 # maxinum control output

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

        plt.xlim(-1.2 * l, 1.2 * l)
        plt.ylim(-1.2 * l, 1.2 * l)
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
    plt.subplot(3, 1, 1)
    plt.plot(t, z[:, 0], color='red', label=r'$\theta$')
    plt.ylabel('angle')
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 2)
    plt.plot(t, z[:, 1], color='blue', label=r'$\omega$')
    plt.xlabel('t')
    plt.ylabel('angular rate')
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 3)
    plt.plot(t, z[:, 2], color='blue', label=r'$\tau$')
    plt.xlabel('t')
    plt.ylabel('torque')
    plt.legend(loc='upper left')

    plt.show()

def controller_pid(l, g, m, theta, omega):
    # Jacobian based angle-torque control
    torque = -l*g*m*sin(theta)

    # PID
    kp = 200
    kd = 2 * np.sqrt(kp)
    q_des = np.pi

    torque = -kp * (theta - q_des) - kd * omega

    return torque

def linearize(goal, params):

    m, l, g, I, b = params.m, params.l, params.g, params.I, params.b

    A = np.array([
        [0, 1],
        [-m*g*l/I * np.cos(goal[0]), -b/I]
    ])
    B = np.array([[0, 1./I]]).T

    return A, B

def lqr_scipy(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    ref: Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = np.array(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the lqr gain
    K = np.array(scipy.linalg.inv(R).dot(B.T.dot(X)))
    eigVals, eigVecs = scipy.linalg.eig(A-B.dot(K))

    return K, X, eigVals

def controller_lqr(K, max_tau, theta, omega):

    goal = [np.pi, 0.0]

    delta_pos = theta - goal[0]
    delta_pos_wrapped = (delta_pos + np.pi) % (2*np.pi) - np.pi
    delta_y = np.asarray([delta_pos_wrapped, omega - goal[1]])

    u = np.asarray(-K.dot(delta_y))[0]

    u = np.clip(u, -max_tau, max_tau)

    return u

def simple_pendulum(z0, t, m, l, c, b, g, max_tau, K):

    theta, omega, tau = z0

    ### Controller here this can be model-based / model-free whatever :)
    
    # PID
    # torque = controller_pid(l, g, m, theta, omega)
    # LQR
    torque = controller_lqr(K, max_tau, theta, omega)

    theta_dd = (torque - m*g*l*sin(theta) - b*omega - np.sign(omega)*c) / (m*l*l)

    return [omega, theta_dd, torque]

if __name__ == '__main__':

    params = Parameters()

    t = np.linspace(0, 5, 500)

    ### LQR
    # Linearize for linear control
    Q = np.diag([10, 1])
    R = np.array([[1]])
    goal = np.array([np.pi, 0])
    A_lin, B_lin = linearize(goal, params)
    K, S, eigVals = lqr_scipy(A_lin, B_lin, Q, R)
    print(f"{K=} {eigVals=}")

    # initlal state
    # z0 = np.array([3.1, 0.001, 10.0])
    z0 = np.array([0.0, 0.001, 0.0])
    all_params = (
        params.m, params.l,
        params.c, params.b,
        params.g, params.max_torque,
        K
    )
    z = odeint(simple_pendulum, z0, t, args=all_params)

    t_interp, z_interp = interpolation(t, z, params)
    animate(t_interp, z_interp, params)
