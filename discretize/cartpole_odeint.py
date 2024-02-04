import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Initialization
tstart = 0
tstop = 2
increment = 0.01

# Initial condition
x_init = np.array([0, 0, np.pi - 0.1, 0])
# x_init = np.array([0, 0, 0, 0])
t = np.arange(tstart,tstop+1,increment)

def sin(theta):
    return np.sin(theta)

def cos(theta):
    return np.cos(theta)

# Function that returns dx/dt
def mydiff(x, t):
    c = 4 # Damping constant
    k = 2 # Stiffness of the spring
    m = 20 # Mass
    F = 5
    dx1dt = x[1]
    dx2dt = (F - c*x[1] - k*x[0])/m
    dxdt = [dx1dt, dx2dt]
    return dxdt

def pendcart(z, t, m, M, L, I, g, d, u):
    
    x, x_dot, theta, theta_dot = z
    
    dx = z[1]
    omega = z[3]

    ax = 1.0*(L**2*g*m**2*sin(2*theta)/2 + (I + L**2*m)*(1.0*L*m*theta_dot**2*sin(theta) - d*x_dot + u))/(I*M + I*m + L**2*M*m + L**2*m**2*sin(theta)**2)
    alpha = 1.0*L*m*(-g*(M + m)*sin(theta) - (1.0*L*m*theta_dot**2*sin(theta) - d*x_dot + u)*cos(theta))/(I*M + I*m + L**2*M*m + L**2*m**2*sin(theta)**2)

    return dx, ax, omega, alpha

# Solve ODE
g = 9.81
M = 0.5
m = 0.2
L = 0.3
I = 0.006
d = 0.1
b = 1 # pendulum up (b=1)
u = 0 # no control
x = odeint(pendcart, x_init, t, args=(m, M, L, I, g, d, u))
# x, x_dot, theta, theta_dot
x1 = x[:,0]
x2 = x[:,1]
x3 = x[:,2]
x4 = x[:,3]

# Plot the Results
plt.plot(t, x1)
plt.plot(t, x2)
plt.plot(t, x3)
plt.plot(t, x4)

plt.title('Simulation of Mass-Spring-Damper System')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend([r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$'])
plt.grid()
plt.show()