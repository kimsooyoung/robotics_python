from matplotlib import pyplot as plt
from scipy.integrate import odeint
import numpy as np


class Parameter():

    def __init__(self):
        self.M = np.array([[1, 0.1], [0.1, 2]])
        self.C = np.array([[0.2, 0], [0, 0.1]])
        self.G = np.array([[5, 1], [1, 10]])

        # self.Kp = 100 * np.identity(2)
        self.Kp = 10 * np.identity(2)
        # self.Kp = 1 * np.identity(2)

        self.Kd = 2 * np.sqrt(self.Kp)

        self.q_des = np.array([0.5, 1.0])

        # self.uncertainty = 0.1
        # self.uncertainty = 0.2
        self.uncertainty = 5.0
        self.M_hat = self.M + self.uncertainty * np.random.rand(2, 2)
        self.C_hat = self.C + self.uncertainty * np.random.rand(2, 2)
        self.G_hat = self.G + self.uncertainty * np.random.rand(2, 2)


def control_partition_rhs(z, t, M, C, K, Kp, Kd, q_des, M_hat, C_hat, G_hat):

    q = np.array([z[0], z[2]])
    q_dot = np.array([z[1], z[3]])

    tau = M_hat@(-Kp@(q-q_des)-Kd@q_dot) + C_hat@(q_dot) + G_hat@(q)

    A = M
    b = tau - (C@q_dot + G@q)
    q_ddot = np.linalg.inv(A) @ b

    return [z[1], q_ddot[0], z[3], q_ddot[1]]


def plot(t, z, params):
    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(t, z[:, 0])
    plt.plot(t, params.q_des[0] * np.ones(len(t)), 'r+')
    plt.xlabel('t')
    plt.ylabel('q1')
    plt.title('Plot of position vs time')

    plt.subplot(2, 1, 2)
    plt.plot(t, z[:, 2])
    plt.plot(t, params.q_des[1] * np.ones(len(t)), 'r+')
    plt.xlabel('t')
    plt.ylabel('q2')

    plt.show()


if __name__ == '__main__':

    params = Parameter()
    q1, q1_dot = 0, 0
    q2, q2_dot = 0, 0

    t0, t_end = 0, 10

    t = np.linspace(t0, t_end, 101)
    z0 = np.array([q1, q1_dot, q2, q2_dot])

    M, C, G = params.M, params.C, params.G
    M_hat, C_hat, G_hat = params.M_hat, params.C_hat, params.G_hat
    Kp, Kd = params.Kp, params.Kd
    q_des = params.q_des

    z = odeint(
        control_partition_rhs, z0, t,
        args=(M, C, G, Kp, Kd, q_des, M_hat, C_hat, G_hat)
    )
    plot(t, z, params)
