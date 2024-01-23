from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint

class Param:

    def __init__(self):
        self.g = -9.81
        self.M = 0.5
        self.m = 0.2
        self.L = 0.3
        self.I = 0.006
        self.d = 0.1
        self.b = 1 # pendulum up (b=1)

        self.pause = 0.005
        self.fps = 10

        # m = 1
        # M = 5
        # L = 2
        # g = -10
        # d = 1

        # b = 1 # pendulum up (b=1)

# x_dot = 0 / theta = 0 / theta_dot = 0
def dynamics(m, M, L, I, d, g):

    b = 1

    A = np.array([
        [0,1,0,0],
        [0,-d/M,b*m*g/M,0],
        [0,0,0,1],
        [0,-b*d/(M*L),-b*(m+M)*g/(M*L),0]
    ])
    B = np.array([0,1/M,0,b/(M*L)]).reshape((4,1))

    return A, B


# def pendcart(x,t, m,M,L,g,d,uf):
def pendcart(x,t, m,M,L,g,d):

    # u = uf(x) # evaluate anonymous function at x
    u = 0

    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m*L*L*(M+m*(1-Cx**2))
    
    dx = np.zeros(4)
    dx[0] = x[1]
    dx[1] = (1/D)*(-(m**2)*(L**2)*g*Cx*Sx + m*(L**2)*(m*L*(x[3]**2)*Sx - d*x[1])) + m*L*L*(1/D)*u
    dx[2] = x[3]
    dx[3] = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*(x[3]**2)*Sx - d*x[1])) - m*L*Cx*(1/D)*u;
    
    return dx

def pendcart_lin(z, t, A, B):

    theta = z[2]

    z = z.reshape((4,1))
    u = np.zeros((1,1))

    result = A @ z + B @ u
    result = result.reshape((4,)).tolist()

    return result


def animate(tspan, x, params):
    
    L = params.L
    W = 0.1
    
    plt.xlim(-2, 2)
    plt.ylim(-0.7, 0.7)
    plt.gca().set_aspect('equal')
    
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Inverted Pendulum')
    
    for i in range(len(tspan)):
        stick, = plt.plot(
            [x[i, 0], x[i, 0] + L*np.sin(x[i, 2])], 
            [0, -L*np.cos(x[i, 2])], 
            'b'
        )
        ball, = plt.plot(
            x[i, 0] + L*np.sin(x[i, 2]), 
            -L*np.cos(x[i, 2]), 
            'ro'
        )
        body, = plt.plot(
            [x[i, 0] - W/2, x[i, 0] + W/2],
            [0, 0],
            linewidth=5,
            color='black'
        )
        
        plt.pause(params.pause)
        stick.remove()
        ball.remove()
        body.remove()
        
    plt.close()


if __name__ == '__main__':

    params = Param()
    m, M, L, I, g, d = params.m, params.M, params.L, params.I, params.g, params.d
    
    ## Simulate closed-loop system
    t0, tend, N = 0, 2, 100
    tspan = np.linspace(t0, tend, N)
    
    # Initial condition (x, x_dot, theta, theta_dot)
    z0 = np.array([0, 0, np.pi - 0.1, 0])

    u = 0
    x = odeint(pendcart, z0, tspan, args=(m, M, L, g, d))
    
    A, B = dynamics(m, M, L, I, d, g)
    x = odeint(pendcart_lin, z0, tspan, args=(A, B))

    print(x)
    animate(tspan, x, params)
