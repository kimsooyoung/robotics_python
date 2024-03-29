import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from spring import spring


class Param:

    def __init__(self):

        self.c = 4 # Damping constant
        self.k = 2 # Stiffness of the spring
        self.m = 20 # Mass
        self.F = 5

        self.pause = 0.01
        self.fps = 10


def dynamics(c, k, m, F):

    A = np.array([
        [0, 1], 
        [-k/m, -c/m]
    ])

    B = np.array([
        [0],
        [1/m]
    ])

    return A, B

# Function that returns dx/dt
def spring_mass_damper_rhs(x, t, A, B):

    x = x.reshape((2,1))
    u = np.zeros((1,1))

    result = A @ x + B @ u

    return result.reshape((2,)).tolist()


def animate(t, result, params):

    x1 = result[:,0]
    x2 = result[:,1]
    ceil = (0.0, 3.0)
    damper_height = 0.5

    plt.xlim(-3, 3)
    plt.ylim(-2.7, 2.7)
    plt.gca().set_aspect('equal')

    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')

    for i in range(len(t)):
        spring_plt, = plt.plot(*spring(ceil, (0, x1[i]), 50, 0.2), c="black")
        damper1, = plt.plot([0, 0], [x1[i], -2], c="black", linewidth=2)
        damper2, = plt.plot([0, 0], [ 0.5*(x1[i]-2), 0.5*(x1[i]-2)-0.2 ], c="black", linewidth=10)
        ball, = plt.plot(x1[i], 'ro', markersize=15)

        plt.pause(params.pause)
        ball.remove()
        spring_plt.remove()
        damper1.remove()
        damper2.remove()

    plt.close()


def plot(t, result):

    # Plot the Results
    x1 = result[:,0]
    x2 = result[:,1]

    plt.plot(t,x1)
    plt.plot(t,x2)

    plt.title('Simulation of Mass-Spring-Damper System')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend(["x1", "x2"])

    plt.grid()
    plt.show()


if __name__ == '__main__':

    params = Param()
    c, k, m, F = params.c, params.k, params.m, params.F
    A, B = dynamics(c, k, m, F)

    # Initialization
    tstart = 0
    tstop = 60
    increment = 0.1

    # Initial condition (x, x_dot)
    x_init = [0, 1]
    t = np.arange(tstart, tstop + 1, increment)

    # Solve ODE
    result = odeint(spring_mass_damper_rhs, x_init, t, args=(A, B))
    
    # visualize
    animate(t, result, params)
    # plot(t, result)
