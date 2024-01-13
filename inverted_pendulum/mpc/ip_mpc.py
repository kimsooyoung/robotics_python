import osqp
import numpy as np
import scipy as sp
import control
from scipy import sparse
from scipy import linalg
from scipy.integrate import odeint
from matplotlib import pyplot as plt

# [x] 일단 다시 짜기
# [x] discretize 하기
# [x] controllability 확인하기
# [] 기존 A, B에 c2d 적용해서 시뮬레이션 해보기
# [] 

class Param:

    def __init__(self):
        # self.g = 9.81
        # self.m = 0.2
        # self.M = 0.5
        # self.L = 0.3
        # self.I = 0.006
        # self.d = 0.1
        # self.b = 1 # pendulum up (b=1)

        # self.g = 9.81
        # self.m = 1
        # self.M = 5
        # self.L = 2
        # self.I = 0
        # self.d = 0.1
        # self.b = 1 # pendulum up (b=1)

        self.M = 0.5  # Cart mass
        self.m = 0.2  # Pendulum mass
        self.b = 0.1  # Coefficient of friction for cart
        self.I = 0.006  # Mass moment of inertia of the pendulum
        self.g = 9.8  # Gravity
        self.l = 0.3  # Length to pendulum center of mass
        self.dt = .1  # Time step

        self.pause = 0.1
        self.fps = 10

def animate(x, params):
    
    m, n = x.shape
    l = params.l
    W = 0.5
    
    # plt.xlim(-50, 50)
    plt.xlim(-10.0, 10.0)
    plt.ylim(-2.7, 2.7)
    plt.gca().set_aspect('equal')
    
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Inverted Pendulum')
    
    for i in range(m):
        stick, = plt.plot(
            [x[i, 0], x[i, 0] + l*np.sin(x[i, 2])], 
            [0, -l*np.cos(x[i, 2])], 
            'b'
        )
        ball, = plt.plot(
            x[i, 0] + l*np.sin(x[i, 2]), 
            -l*np.cos(x[i, 2]), 
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

param = Param()
M, m, l, I, g, b, dt = param.M, param.m, param.l, param.I, param.g, param.b, param.dt

p = I*(M+m)+M*m*l**2

A = np.array([[0,      1,              0,            0],
              [0, -(I+m*l**2)*b/p,  (m**2*g*l**2)/p, 0],
              [0,      0,              0,            1],
              [0, -(m*l*b)/p,       m*g*l*(M+m)/p,   0]])

B = np.array([[0],
              [(I+m*l**2)/p],
              [0],
              [m*l/p]])

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

D = np.array([[0],
              [0]])

sys = control.StateSpace(A, B, C, D)
sys_discrete = control.c2d(sys, dt, method='zoh')

Ad = np.array(sys_discrete.A)
Bd = np.array(sys_discrete.B)
[nx, nu] = Bd.shape
print(f"nx: {nx}, nu: {nu}")

# #################################################################

# param = Param()
# m, M, L, d, g = param.m, param.M, param.L, param.d, param.g

# # Discrete time model of a inverted pendulum
# Ad = np.array([
#     [0, 1, 0, 0], 
#     [0, -1.0*d/M, 1.0*g*m/M, 0], 
#     [0, 0, 0, 1], 
#     [0, -1.0*d/(L*M), 1.0*g*(M + m)/(L*M), 0]
# ])
# Bd = np.array([
#     [0], 
#     [1/M], 
#     [0], 
#     [1/(M*L)]
# ])
# [nx, nu] = Bd.shape
# # print(f"nx: {nx}, nu: {nu}")
# #################################################################


# Constraints
u_abs = 500.0
umin = np.array([-u_abs])
umax = np.array([+u_abs])
xmin = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
xmax = np.array([+np.inf, +np.inf, +np.inf,  np.inf])

# Objective function
Q = sparse.diags([1., 1., 10., 1.])
QN = Q
R = 0.1*sparse.eye(1)

# Initial and reference states
x0 = np.array([0., 0., np.pi, 0.])
xr = np.array([0.1, 0., np.pi, 0.])

# Prediction horizon
N = 20

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective (Hessian)
# Q_np = np.diag([1., 1., 1., 1.])
# QN_np = Q_np
# R_np = 0.1 * np.eye(4)
# temp = [np.kron(np.eye(N), Q_np), QN_np, np.kron(np.eye(N), R_np)]
# P_np = linalg.block_diag([np.kron(np.eye(N), Q_np), QN_np,
#                        np.kron(np.eye(N), R_np)])

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

# Simulate in closed loop
nsim = 200
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
    x0 = Ad@x0 + Bd@ctrl
    print(x0)

    state[i+1] = x0

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)

animate(state, param)