import osqp
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.integrate import odeint
from matplotlib import pyplot as plt

class Param:

    def __init__(self):
        self.g = 9.81
        self.m = 1
        self.M = 5
        self.L = 2
        self.d = 0
        self.b = 1 # pendulum up (b=1)

        self.pause = 0.05
        self.fps = 10

# x_dot = 0 / theta = sy.pi / theta_dot = 0
def dynamics(m, M, L, d, g):

    A = np.array([
        [0, 1, 0, 0], 
        [0, -1.0*d/M, 1.0*g*m/M, 0], 
        [0, 0, 0, 1], 
        [0, -1.0*d/(L*M), 1.0*g*(M + m)/(L*M), 0]
    ])

    B = np.array([0, 1/M, 0, 1/(M*L)]).reshape((4,1))

    return A, B

def pendcart_non_linear(z, t, m, M, L, g, d, u):

    x, x_dot, theta, theta_dot = z
    
    dx = z[1]
    ax = 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + g*m*np.sin(2*theta)/2 + u)/(M + m*np.sin(theta)**2)
    omega = z[3]
    alpha = -(1.0*g*(M + m)*np.sin(theta) + 1.0*(1.0*L*m*theta_dot**2*np.sin(theta) - d*x_dot + u)*np.cos(theta))/(L*(M + m*np.sin(theta)**2))

    return dx, ax, omega, alpha

def animate(x, params):
    
    m, n = x.shape
    L = params.L
    W = 0.5
    
    # plt.xlim(-50, 50)
    plt.xlim(-5, 5)
    plt.ylim(-2.7, 2.7)
    plt.gca().set_aspect('equal')
    
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Inverted Pendulum')
    
    for i in range(m):
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
    m, M, L, d, g = params.m, params.M, params.L, params.d, params.g

    # Discrete time model of a inverted pendulum
    A, B = dynamics(m, M, L, d, g)

    Ad = sparse.csc_matrix(A)
    Bd = sparse.csc_matrix(B)
    [nx, nu] = Bd.shape

    # Constraints
    u0 = 0.0
    umin = np.array([-300.0])
    umax = np.array([+300.0])
    xmin = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
    xmax = np.array([+np.inf,  np.inf, +np.inf,  np.inf])

    # Objective function
    Q = sparse.diags([1., 1., 1., 1.])
    QN = Q
    R = 1*sparse.eye(1)

    # Initial and reference states
    x0 = np.array([-1, 0, np.pi+0.1, 0])
    xr = np.array([1, 0, np.pi, 0])

    # Prediction horizon
    N = 20

    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # - quadratic objective (Hessian)
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                        sparse.kron(sparse.eye(N), R)], format='csc')

    # - linear objective (Gradient)
    q = np.hstack([np.kron(np.ones(N), -Q@xr), -QN@xr, np.zeros(N*nu)])

    # - linear dynamics
    Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
    Aeq = sparse.hstack([Ax, Bu])
    leq = np.hstack([-x0, np.zeros(N*nx)])
    ueq = leq

    # - input and state constraints
    Aineq = sparse.eye((N+1)*nx + N*nu)
    lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
    # - OSQP constraints
    A = sparse.vstack([Aeq, Aineq], format='csc')
    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u, verbose=False)

    # Simulate and solve
    nsim = 30
    state = np.zeros((nsim+1, nx))
    state[0] = x0
    for i in range(nsim):
        # Solve
        res = prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # Apply first control input to the plant
        ctrl = res.x[-N*nu:-(N-1)*nu]
        # print(ctrl)
        # z_result = odeint(pendcart_non_linear, x0, tspan, args=(m, M, L, g, d, K2, z_ref))

        # Parse state
        x0 = Ad@x0 + Bd@ctrl
        # print(x0)
        state[i+1] = x0

        # Update initial state
        l[:nx] = -x0
        u[:nx] = -x0
        prob.update(l=l, u=u)

    print(state)
    animate(state, params)
