import osqp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from spring import spring

import control
from scipy import sparse

class Param:

    def __init__(self):

        self.c = 4 # Damping constant
        self.k = 2 # Stiffness of the spring
        self.m = 20 # Mass
        self.F = 5

        # Objective function
        self.Q = sparse.diags([10.0, 0.0])
        self.R = 0.1 * sparse.eye(1)

        self.umin = -500.0
        self.umax = 500.0
        self.N = 10

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


# ref from : https://osqp.org/docs/examples/mpc.html
def qp_mpc(u_min, u_max, x0, xr, N, Ad, Bd, Q, R):

    QN = Q
    [nx, nu] = Bd.shape

    # Constraints
    u0 = -150.0
    umin = np.array([u_min])
    umax = np.array([u_max])
    xmin = np.array([-np.inf, -np.inf])
    xmax = np.array([+np.inf, +np.inf])

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


# Function that returns dx/dt
def spring_mass_damper_rhs(x, t, A, B, u):

    x = x.reshape((2,1))
    u = u.reshape((1,1))

    result = A@x + B@u

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


def plot(t, result, u_list=None):

    # Plot the Results
    x1 = result[:,0]
    x2 = result[:,1]

    plt.subplot(2, 1, 1)
    plt.plot(t,x1)
    plt.plot(t,x2)
    plt.title('Simulation of Mass-Spring-Damper System')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend(["x1", "x2"])
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(u_list, color='brown', label=r'$u$')
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == '__main__':

    # Define the parameters
    params = Param()
    c, k, m, F = params.c, params.k, params.m, params.F
    umin, umax, N = params.umin, params.umax, params.N
    Q, R = params.Q, params.R

    # Define the system
    A, B = dynamics(c, k, m, F)
    C_o = control.ctrb(A, B)
    print(A, B)
    print(f'C_o rank: {np.linalg.matrix_rank(C_o)}')

    # Convert to sparse matrix
    Ad = sparse.csc_matrix(A)
    Bd = sparse.csc_matrix(B)
    [nx, nu] = Bd.shape
    print(nx, nu)

    # Initial and Final condition (x, x_dot)
    x_init = np.array([0., 1.])
    x_ref  = np.array([-1., 0.])

    # Prepare for MPC
    P, q, A, l, u = qp_mpc(umin, umax, x_init, x_ref, N, Ad, Bd, Q, R)
    print(f'P: {P.shape}, q: {q.shape}, A: {A.shape}, l: {l.shape}, u: {u.shape}')

    # Create an OSQP object and Setup
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, verbose=False)

    # Simulate and solve
    nsim = 30
    t0, tend = 0, 10
    t_span = np.linspace(t0, tend, nsim + 1)

    # Define result holders
    x0 = x_init
    state_holder = np.zeros((nsim+1, nx))
    state_holder[0] = x0
    control_holder = np.zeros((nsim+1, 1))
    control_holder[0] = np.zeros(1)

    for i in range(nsim):
        # Solve
        res = prob.solve()
        print(f"res.x : {res.x}")

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # Apply first control input to the plant
        ctrl = res.x[-N*nu:-(N-1)*nu]
        control_holder[i] = ctrl
        print(f"prev state / ctrl : {x0} / {ctrl}")
        
        # Parse state
        x0 = Ad@x0 + Bd@ctrl

        # t_temp = np.array([t_span[i], t_span[i+1]])
        # z_result = odeint(spring_mass_damper_rhs, x0, t_temp, args=(A, B, ctrl))
        # x0 = z_result[1]

        print(f"new state : {x0}")

        # Update state
        state_holder[i+1] = x0
        l[:nx] = -x0
        u[:nx] = -x0
        prob.update(l=l, u=u)

    # # Solve ODE
    # force_result = []
    # result = odeint(spring_mass_damper_rhs, x_init, t, args=(A, B, K, x_ref, force_result))
    
    # # visualize
    # animate(t_span, state_holder, params)
    plot(t_span, state_holder, control_holder)
