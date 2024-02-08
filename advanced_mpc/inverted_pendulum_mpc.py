import osqp
import control
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.integrate import odeint
from matplotlib import pyplot as plt

def sin(theta):
    return np.sin(theta)

def cos(theta):
    return np.cos(theta)

class Param:

    def __init__(self):

        self.g = 9.81
        self.M = 0.5
        self.m = 0.2
        self.L = 0.3
        self.I = 0.006
        self.d = 0.1
        self.b = 1 # pendulum up (b=1)

        # Objective function
        self.Q = sparse.diags([100., 10., 100., 10.])
        self.R = 0.1 * sparse.eye(1)
        self.increment = 0.1
        self.umin = -500.0
        self.umax = 500.0
        self.N = 20

        self.pause = 0.1
        self.fps = 10

# x_dot = 0 / theta = sy.pi / theta_dot = 0
def dynamics(m, M, L, I, d, g):
    # ref : https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling

    A = np.array([
        [0, 1, 0, 0], 
        [0, -1.0*d*(I + L**2*m)/(I*M + I*m + L**2*M*m), 1.0*L**2*g*m**2/(I*M + I*m + L**2*M*m), 0], 
        [0, 0, 0, 1], 
        [0, 1.0*L*d*m/(I*M + I*m + L**2*M*m), -1.0*L*g*m*(M + m)/(I*M + I*m + L**2*M*m), 0]
    ])

    B = np.array([
        [0], 
        [1.0*(I + L**2*m)/(I*M + I*m + L**2*M*m)], 
        [0], 
        [-1.0*L*m/(I*M + I*m + L**2*M*m)]
    ])

    C = np.array([[1, 0, 0, 0]])

    return A, B, C

def pendcart_non_linear(z, t, m, M, L, g, d, u):

    x, x_dot, theta, theta_dot = z

    dx = z[1]
    omega = z[3]
    ax = 1.0*(L**2*g*m**2*sin(2*theta)/2 + (I + L**2*m)*(1.0*L*m*theta_dot**2*sin(theta) - d*x_dot + u))/(I*M + I*m + L**2*M*m + L**2*m**2*sin(theta)**2)
    alpha = 1.0*L*m*(-g*(M + m)*sin(theta) - (1.0*L*m*theta_dot**2*sin(theta) - d*x_dot + u)*cos(theta))/(I*M + I*m + L**2*M*m + L**2*m**2*sin(theta)**2)

    return dx, ax, omega, alpha


def qp_mpc(u_min, u_max, x0, xr, N, Ad, Bd, Q, R):

    QN = Q
    [nx, nu] = Bd.shape

    # Constraints
    u0 = 0.0
    umin = np.array([u_min])
    umax = np.array([u_max])
    xmin = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
    xmax = np.array([+np.inf, +np.inf, +np.inf,  np.inf])

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

    return P, q, A, l, u


def animate(x, params):
    
    m, n = x.shape
    L = params.L
    W = 0.1
    
    # plt.xlim(-50, 50)
    plt.xlim(-2.0, 2.0)
    plt.ylim(-0.7, 0.7)
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

def plot(t, result, control):

    x, x_dot, theta, theta_dot = result.T

    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(t, x, label='x')
    plt.plot(t, theta, label='theta')
    plt.plot(t, x_dot, label='x_dot')
    plt.plot(t, theta_dot, label='theta_dot')
    plt.legend([r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$'])
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t, control, label='u')
    plt.grid()

    plt.show()

if __name__ == '__main__':

    params = Param()
    m, M, L, I, g, d = params.m, params.M, params.L, params.I, params.g, params.d
    increment, umin, umax, N = params.increment, params.umin, params.umax, params.N
    Q, R = params.Q, params.R

    # Discrete time model of a inverted pendulum
    A, B, C = dynamics(m, M, L, I, d, g)
    C_o = control.ctrb(A, B)
    print(A, B)
    print(f'C_o rank: {np.linalg.matrix_rank(C_o)}')

    # Discretize the system
    sys = control.ss(A, B, C, 0)
    sysDisc = control.sample_system(sys, increment, method='zoh') 
    print(sysDisc.A, sysDisc.B)
    
    # Swap the system matrices
    A, B = sysDisc.A, sysDisc.B

    Ad = sparse.csc_matrix(A)
    Bd = sparse.csc_matrix(B)
    [nx, nu] = Bd.shape

    # Initial and reference states
    # x0 = np.array([-1, 0, np.pi+0.1, 0])
    x0 = np.array([0, 0, np.pi, 0])
    xr = np.array([1, 0, np.pi, 0])

    # Prediction horizon
    P, q, A, l, u = qp_mpc(umin, umax, x0, xr, N, Ad, Bd, Q, R)

    # Create an OSQP object and Setup
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, verbose=False)

    # Simulate and solve
    tstart = 0
    tstop = 10
    t_span = np.arange(tstart, tstop, increment)
    nsim = len(t_span) - 1

    # Define result holders
    state_holder = np.zeros((nsim+1, nx))
    state_holder[0] = x0
    control_holder = np.zeros((nsim+1, 1))
    control_holder[0] = np.zeros(1)

    for i in range(nsim):
        # Solve
        res = prob.solve()
        # print(f"res.x : {res.x}")

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # Apply first control input to the plant
        ctrl = res.x[-N*nu:-(N-1)*nu]
        control_holder[i] = ctrl
        # print(f"ctrl : {ctrl}")
        
        # t_temp = np.array([t_span[i], t_span[i+1]])
        # z_result = odeint(pendcart_non_linear, x0, t_temp, args=(m, M, L, g, d, ctrl[0]))
        # x0 = z_result[1]
        
        # Parse state
        x0 = Ad@x0 + Bd@ctrl

        # Update state
        state_holder[i+1] = x0
        l[:nx] = -x0
        u[:nx] = -x0
        prob.update(l=l, u=u)

    animate(state_holder, params)
    # plot(t_span, state_holder, control_holder)
