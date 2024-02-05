# ref : https://www.halvorsen.blog/documents/programming/python/resources/powerpoints/Discrete%20Systems%20with%20Python.pdf
# recap spring mass damper system
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class Param:

    def __init__(self):

        self.c = 4 # Damping constant
        self.k = 2 # Stiffness of the spring
        self.m = 20 # Mass
        self.F = 5

        self.pause = 0.1
        self.fps = 10


# Function that returns dx/dt
def spring_mass_damper_rhs(x, t, c, k, m, F):

    x_dot = x[1]
    x_ddot = (F - c*x[1] - k*x[0])/m

    return x_dot, x_ddot


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

    # Initialization
    tstart = 0
    tstop = 60
    increment = 0.1

    # Initial condition (x, x_dot)
    x_init = [0, 1]
    t = np.arange(tstart, tstop + 1, increment)

    # Solve ODE
    result = odeint(spring_mass_damper_rhs, x_init, t, args=(c, k, m, F))
    plot(t, result)
