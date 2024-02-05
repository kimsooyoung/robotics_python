# ref : https://www.halvorsen.blog/documents/programming/python/resources/powerpoints/Discrete%20Systems%20with%20Python.pdf

# Simulation of Mass-Spring-Damper System
import numpy as np
import matplotlib.pyplot as plt


class Param:

    def __init__(self):

        self.c = 4 # Damping constant
        self.k = 2 # Stiffness of the spring
        self.m = 20 # Mass
        self.F = 5

        self.pause = 0.1
        self.fps = 10


def plot(t, x1, x2):

    # Plot the Results
    plt.plot(t, x1)
    plt.plot(t, x2)

    plt.title('Simulation of Mass-Spring-Damper System')
    plt.xlabel('t[s]')
    plt.ylabel('x(t)')
    
    plt.grid()
    plt.legend(["x1", "x2"])
    plt.show()


if __name__ == '__main__':

    # Dynamics Parameters
    params = Param()
    c, k, m, F = params.c, params.k, params.m, params.F

    # Simulation Parameters
    tstart = 0
    tstop = 60
    increment = 0.1

    N = int((tstop - tstart) / increment) # Simulation length
    x1, x2 = np.zeros(N+2), np.zeros(N+2)
    x1[0], x2[0] = 0, 1 # Initial Position, Initial Speed

    a11, a12 = 1, increment
    a21, a22 = -(increment * k) / m, 1 - (increment * c) / m
    b1, b2 = 0, increment / m

    # Simulation
    for k in range(N+1):
        x1[k+1] = a11 * x1[k] + a12 * x2[k] + b1 * F
        x2[k+1] = a21 * x1[k] + a22 * x2[k] + b2 * F

    # Plot the Simulation Results
    t = np.arange(tstart, tstop + 2 * increment, increment)
    plot(t, x1, x2)
