import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np


class Parameters:
    def __init__(self):
        # D : distance between start and end
        # N : number of collocation points
        self.D1 = 0.5
        self.V1 = 0.2
        
        self.D2 = 1
        self.V2 = 0
        
        self.N = 20

def nonlinear_func(x, phase=1):
    
    params = Parameters()
    
    N = params.N
    T = x[0]
    # N points means N+1 steps
    t = np.linspace(0, T, N+1)
    dt = t[1] - t[0]
    
    pos = np.zeros(N+1)
    vel = np.zeros(N+1)
    u   = np.zeros(N+1)
    
    # seperate x vals into pos, vel, u
    # i: 0 ~ N
    for i in range(N+1):
        # x[1] ~ x[N+1] : pos
        pos[i] = x[i+1]
        # x[N+2] ~ x[2N+2] : vel
        vel[i] = x[i+N+2]
        # x[2N+3] ~ x[3N+4] : u
        u[i]   = x[i+2*N+3]
    
    # prepare dynamics equations
    defect_pos = np.zeros(N)
    defect_vel = np.zeros(N)
    for i in range(N):
        defect_pos[i] = pos[i+1] - pos[i] - dt*vel[i]
        defect_vel[i] = vel[i+1] - vel[i] - dt*0.5*(u[i]+u[i+1])
    
    # pos dynamics eq N ea
    # vel dynamics eq N ea
    # pos start, end cond
    # vel start, end cond
    # acc start cond 
    #     => total 2N + 5 ea
    ceq = np.zeros(2*N + 5)
    
    # pos(0) = 0, pos(N) = D
    # vel(0) = 0, vel(N) = 0
    if phase == 1:
        ceq[0] = pos[0]
        ceq[1] = vel[0]
        ceq[2] = pos[N] - params.D1
        ceq[3] = vel[N] - params.V1
        ceq[4] = u[0]
    elif phase == 2:
        ceq[0] = pos[0] - params.D1
        ceq[1] = vel[0] - params.V1
        ceq[2] = pos[N] - params.D2
        ceq[3] = vel[N] - params.V2
        ceq[4] = u[-1]
    
    # dynamics eq
    for i in range(N):
        ceq[i+5] = defect_pos[i]
        ceq[i+N+5] = defect_vel[i]
    
    return ceq
        
def nonlinear_func1(x):
    return nonlinear_func(x, 1)

def nonlinear_func2(x):
    return nonlinear_func(x, 1)

x0 = np.zeros(3*20 + 4)

print(nonlinear_func(x0, 1))