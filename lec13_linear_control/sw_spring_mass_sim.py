from matplotlib import pyplot as plt
import numpy as np
import control
from scipy.integrate import odeint

class Parameters():

    def __init__(self) -> None:
        
        self.m1, self.m2 = 1, 1
        self.k1, self.k2 = 2, 3

        self.Q = np.eye(4)
        self.R = 0.01 * np.eye(2)

        self.pause = 0.01

def dynamisc(m1, m2, k1, k2):

    A = np.array([
        [0,0,1,0],
        [0,0,0,1],
        [-(k1/m1+k2/m1), k2/m1, 0, 0],
        [k2/m2, -k2/m2, 0, 0]
    ])
    
    B = np.array([
        [0,0],
        [0,0],
        [-1/m1, 0],
        [1/m2,1/m2]
    ])

    return A, B

def spring_mass_linear_equ(x, t, m1, m2, k1, k2, K):

    A, B = dynamisc(m1, m2, k1, k2)
    u = -K@x

    return A@x + B@u

def plot(result, ts):

    plt.figure(1)
    
    plt.subplot(2,1,1)
    plt.plot(ts, result[:,0],'r-.')
    plt.plot(ts, result[:,1],'b');
    plt.ylabel("position")
    
    plt.subplot(2,1,2)
    plt.plot(ts, result[:,2],'r-.')
    plt.plot(ts, result[:,3],'b');
    plt.ylabel("velocity")
    plt.show()

if __name__=="__main__":

    params = Parameters()

    m1, m2, k1, k2 = params.m1, params.m2, params.k1, params.k2
    Q, R = params.Q, params.R

    A, B = dynamisc(m1, m2, k1, k2)
    K, S, E = control.lqr(A, B, Q, R)

    t0, tend = 0, 10
    ts = np.linspace(t0, tend, 101)

    z0 = np.array([0.5,0,0,0])

    result = odeint(spring_mass_linear_equ, z0, ts, args=(m1, m2, k1, k2, K))

    plot(result, ts)