from matplotlib import pyplot as plt
import numpy as np

from scipy.integrate import odeint
from scipy import interpolate


class Parameters():

    def __init__(self):

        self.m = 1
        self.l = 1
        self.g = 9.81

        self.I = 1/12 * (self.m * self.l**2)

        self.kp = 100
        # kd는 운동 방정식 구한 후에 계산
        # 그런데 운동방정식이 비선형이라... 근의 공식이 안됨
        # cos을 테일러 전개하면 첫 항이 1이므로, k=0이라고 해보자.
        self.kd = 2 * np.sqrt(self.kp)

        self.q_des = np.pi / 2

        self.pause = 0.001
        self.fps = 30

        self.tau_noise_mean, self.tau_noise_std = 0, 0.1 * 20
        self.theta_noise_mean, self.theta_noise_std = 0, 0.01 * 10
        self.omega_noise_mean, self.omega_noise_std = 0, 0.1 * 1.0


def get_tau(theta, omega, kp, kd, q_des):

    return -kp * (theta - q_des) - kd * omega


def one_link_manipulator(q0, t, m, l, g, I, kp, kd, q_des, tau_disturb):

    theta, omega = q0
    tau = get_tau(theta, omega, kp, kd, q_des) - tau_disturb

    angular_acc = ( tau - (m*g*l*np.cos(theta))/2 ) / (I + m*l**2/4 )
    return np.array([omega, angular_acc])


def animate(t, z, params):

    t_anim = np.arange(t[0], t[-1], 1/params.fps)
    m, n = np.shape(z)
    z_anim = np.zeros((len(t_anim), n))

    for i in range(n):
        f = interpolate.interp1d(t, z[:,i])
        z_anim[:,i] = f(t_anim)

    l = params.l

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal')

    for i in range(len(t_anim)):
        theta = z_anim[i,0]
        O = np.array([0, 0])
        P = np.array([l*np.cos(theta), l*np.sin(theta)])

        pendulum, = plt.plot([O[0], P[0]], [O[1], P[1]], linewidth=5, color='red')

        plt.pause(params.pause)
        pendulum.remove()

    plt.close()


def plot(t, z, tau, params):

    plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.plot(t, params.q_des * np.ones(len(t)), 'r-.')
    plt.plot(t, z[:, 0])
    plt.ylabel('theta1')
    plt.title('Plot of position, velocity, and Torque vs. time')

    plt.subplot(3, 1, 2)
    plt.plot(t, z[:, 1])
    plt.ylabel('theta1dot')

    plt.subplot(3, 1, 3)
    plt.plot(t, tau[:, 0])
    plt.xlabel('t')
    plt.ylabel('Torque')

    plt.show()    


if __name__=='__main__':

    params = Parameters()

    theta0, omega0 = 0, 0
    t0, t_end = 0, 2

    z0 = np.array([theta0, omega0])

    N = 100
    t = np.linspace(t0, t_end, N)

    m, l, g, I = params.m, params.l, params.g, params.I
    kp, kd, q_des = params.kp, params.kd, params.q_des
    t_mean, t_std = params.tau_noise_mean, params.tau_noise_std
    theta_mean, theta_std = params.theta_noise_mean, params.theta_noise_std
    omega_mean, omega_std = params.omega_noise_mean, params.omega_noise_std

    z = np.zeros((N, 2))
    tau = np.zeros((N, 1))

    z[0] = z0
    tau[0] = 0

    for i in range(len(t)-1):

        time_temp = np.array([t[i], t[i+1]])
        tau_disturb = np.random.normal(t_mean, t_std)

        result = odeint(
            one_link_manipulator, z0, time_temp, args=(m, l, g, I, kp, kd, q_des, tau_disturb)
        )

        z0 = np.array([
            result[1, 0] + np.random.normal(theta_mean, theta_std),
            result[1, 1] + np.random.normal(omega_mean, omega_std)
        ])

        z[i+1] = z0
        tau[i+1] = get_tau(z0[0], z0[1], kp, kd, q_des)

    # animate(t, z, params)
    plot(t, z, tau, params)
