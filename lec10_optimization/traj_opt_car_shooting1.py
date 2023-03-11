from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import interpolate
from scipy.integrate import odeint
import random
import scipy.optimize as opt

class parameters:
    def __init__(self):
        self.D = 5
        self.N = 5
        self.z0 = np.array([0,0])

def car(z, t, t1,t2,u1,u2):

    xdot = z[1];

    t_opt = np.array([t1,t2])
    u_opt = np.array([u1,u2])
    if (t>t2 or t<= t1): #needed because odeint goes outside t bounds
        u = 0
    else:
        f = interpolate.interp1d(t_opt, u_opt)
        u = f(t)

    dzdt = np.array([xdot, u]);
    return dzdt

def simulator(x,z0,N):
    T = x[0];
    for i in range(0,N+1):
        u_opt[i] = x[i+1]

    t_opt = np.linspace(0,T,N+1)

    shape = (N+1,2)
    zz = np.zeros(shape)
    zz[0,0] = z0[0]
    zz[0,1] = z0[1]
    tt = t_opt
    uu = u_opt
    for i in range(0,N): #goes from 0 to N
        args = (t_opt[i],t_opt[i+1],u_opt[i],u_opt[i+1])
        z = odeint(car, z0, np.array([t_opt[i],t_opt[i+1]]), args,rtol=1e-12,atol=1e-12)
        z0 = np.array([z[1,0], z[1,1]]);
        zz[i+1,0] = z0[0]
        zz[i+1,1] = z0[1]

    return tt,zz,uu

def cost(x):
    return x[0]

def nonlinear_fn(x):
    parms = parameters()
    D = parms.D
    z0 = parms.z0
    N = parms.N

    [tt,zz,uu] = simulator(x,z0,N);
    x_end = zz[N,0]-D;
    v_end = zz[N,1];
    return [x_end, v_end ]

random.seed(1)
parms = parameters()
N = parms.N

T_min = 1; T_max = 5;
u_min = -5; u_max = 5;

T_opt = 2
u_opt = [0] * (N+1) #initialize u to zeros
for i in range(0,N+1):
    u_opt[i] = u_min + (u_max-u_min)*random.random();

x0 = [0] * (N+2)
x_min = [0] * (N+2)
x_max = [0] * (N+2)
x0[0] = T_opt
x_min[0] = T_min
x_max[0] = T_max
for i in range(1,N+2):
    x0[i] = u_opt[i-1]
    x_min[i] = u_min
    x_max[i] = u_max

limits = opt.Bounds(x_min,
                    x_max)

eq_cons = {'type': 'eq',
           'fun' : nonlinear_fn
           }

res = opt.minimize(cost, x0, method='SLSQP',
               constraints=[eq_cons], options={'ftol': 1e-6, 'disp': True, 'maxiter':500},
               bounds=limits)

print(x0)
x = res.x
print(x)
print(res.success)
print(res.message)
print(res.status)

#testing
z0 = parms.z0
[tt,zz,uu] = simulator(x,z0,N) #test
print(zz[N,0])
print(zz[N,1])

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(tt,zz[:,0])
plt.ylabel("x")
plt.subplot(3,1,2)
plt.plot(tt,zz[:,1])
plt.ylabel("xdot")
plt.subplot(3,1,3)
plt.plot(tt,uu)
plt.xlabel("t")
plt.ylabel("u")
# plt.show()
plt.show(block=False)
plt.pause(10)
plt.close()
