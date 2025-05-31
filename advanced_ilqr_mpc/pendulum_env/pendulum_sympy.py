"""
Pendulum Dynamics
=================
"""

import numpy as np
import sympy as smp

try:
    import pydrake.symbolic as sym
    pydrake_available = True
except ModuleNotFoundError:
    pydrake_available = False
    sym = None


def check_type(x):
    """
    checks the type of x and returns the suitable library
    (pydrake.symbolic, sympy or numpy) for furhter calculations on x.
    """
    if isinstance(x, (tuple, np.ndarray)) and isinstance(x[0], smp.Expr):
        md = smp
    elif isinstance(x, np.ndarray) and x.dtype == object and pydrake_available:
        md = sym
    else:
        md = np
    return md


def pendulum_continuous_dynamics(x, u, m=0.5, l=0.5,
                                 b=0.15, cf=0.0, g=9.81, inertia=0.125):
    md = check_type(x)

    pos = x[0]
    vel = x[1]
    if md == smp or md == sym:
        accn = (u[0] - m * g * l * md.sin(pos) - b * vel -
                cf*md.atan(1e8*vel)*2/np.pi) / inertia
    elif md == np:
        accn = (u[0] - m * g * l * md.sin(pos) - b * vel -
                cf*md.arctan(1e8*vel)*2/np.pi) / inertia
    xd = np.array([vel, accn])
    return xd


def pendulum_discrete_dynamics_euler(x, u, dt, m=0.57288, l=0.5, b=0.15,
                                     cf=0.0, g=9.81, inertia=0.125):
    md = check_type(x)

    x_d = pendulum_continuous_dynamics(x, u, m=m, l=l, b=b, cf=cf,
                                       g=g, inertia=inertia)
    x_next = x + x_d*dt
    if md == smp:
        x_next = tuple(x_next)
    return x_next


def pendulum_discrete_dynamics_rungekutta(x, u, dt, m=0.5, l=0.5,
                                          b=0.15, cf=0.0, g=9.81,
                                          inertia=0.125):
    md = check_type(x)

    k1 = pendulum_continuous_dynamics(x, u, m=m, l=l, b=b, cf=cf,
                                      g=g, inertia=inertia)
    k2 = pendulum_continuous_dynamics(x+0.5*dt*k1, u, m=m, l=l, b=b,
                                      cf=cf, g=g, inertia=inertia)
    k3 = pendulum_continuous_dynamics(x+0.5*dt*k2, u, m=m, l=l, b=b,
                                      cf=cf, g=g, inertia=inertia)
    k4 = pendulum_continuous_dynamics(x+dt*k3, u, m=m, l=l, b=b,
                                      cf=cf, g=g, inertia=inertia)
    x_d = (k1 + 2 * (k2 + k3) + k4) / 6.0
    x_next = x + x_d*dt
    if md == smp:
        x_next = tuple(x_next)
    return x_next


def pendulum_swingup_stage_cost(x, u, goal=[np.pi, 0], Cu=10.0, Cp=0.01,
                                Cv=0.01, Cen=0.0, m=0.5, l=0.5, b=0.15,
                                cf=0.0, g=9.81):
    md = check_type(x)

    eps = 1e-6
    c_pos = (x[0] - goal[0] + eps)**2.0
    c_vel = (x[1] - goal[1] + eps)**2.0
    c_control = u[0]**2
    en_g = 0.5*m*(l*goal[1])**2.0 + m*g*l*(1.0-md.cos(goal[0]))
    en = 0.5*m*(l*x[1])**2.0 + m*g*l*(1.0-md.cos(x[0]))
    c_en = (en-en_g+eps)**2.0
    return Cu*c_control + Cp*c_pos + Cv*c_vel + Cen*c_en


def pendulum_swingup_final_cost(x, goal=[np.pi, 0], Cp=1000.0, Cv=10.0,
                                Cen=0.0, m=0.5, l=0.5, b=0.15,
                                cf=0.0, g=9.81):
    md = check_type(x)

    eps = 1e-6
    c_pos = (x[0] - goal[0] + eps)**2.0
    c_vel = (x[1] - goal[1] + eps)**2.0
    en_g = 0.5*m*(l*goal[1])**2.0 + m*g*l*(1.0-md.cos(goal[0]))
    en = 0.5*m*(l*x[1])**2.0 + m*g*l*(1.0-md.cos(x[0]))
    c_en = (en-en_g+eps)**2.0
    return Cp*c_pos + Cv*c_vel + Cen*c_en


def pendulum3_discrete_dynamics_euler(x, u, dt, m=0.5, l=0.5, b=0.15,
                                      cf=0.0, g=9.81, inertia=0.125):
    # pendulum state x = [cos(theta), sin(theta), thetadot]
    md = check_type(x)

    if md == np:
        x2 = np.array([np.arctan2(x[1], x[0]), x[2]])
    if md == smp or md == sym:
        x2 = np.array([md.atan2(x[1], x[0]), x[2]])
    x_next = pendulum_discrete_dynamics_euler(x2, u, dt, m=m, l=l,
                                              b=b, cf=cf, g=g, inertia=inertia)
    x3 = np.array([md.cos(x_next[0]), md.sin(x_next[0]), x_next[1]])
    if md == smp:
        x3 = tuple(x3)
    return x3


def pendulum3_discrete_dynamics_rungekutta(x, u, dt, m=0.5, l=0.5, b=0.15,
                                           cf=0.0, g=9.81, inertia=0.125):
    # pendulum state x = [cos(theta), sin(theta), thetadot]
    md = check_type(x)

    if md == np:
        x2 = np.array([np.arctan2(x[1], x[0]), x[2]])
    if md == smp or md == sym:
        x2 = np.array([md.atan2(x[1], x[0]), x[2]])
    x_next = pendulum_discrete_dynamics_rungekutta(x2, u, dt, m=m,
                                                   l=l, b=b, cf=cf, g=g,
                                                   inertia=inertia)
    x3 = np.array([md.cos(x_next[0]), md.sin(x_next[0]), x_next[1]])
    if md == smp:
        x3 = tuple(x3)
    return x3


def pendulum3_swingup_stage_cost(x, u, goal=[-1, 0, 0], Cu=10.0, Cp=0.01,
                                 Cv=0.01, Cen=0.0, m=0.5, l=0.5, b=0.15,
                                 cf=0.0, g=9.81):
    # pendulum state x = [cos(theta), sin(theta), thetadot]
    eps = 1e-6
    c_pos1 = (x[0] - goal[0] + eps)**2.0
    c_pos2 = (x[1] - goal[1] + eps)**2.0
    c_vel = (x[2] - goal[2] + eps)**2.0
    c_control = u[0]**2
    en_g = 0.5*m*(l*goal[2])**2.0 + m*g*l*(1.0-goal[0])
    en = 0.5*m*(l*x[2])**2.0 + m*g*l*(1.0-x[0])
    c_en = (en-en_g+eps)**2.0
    return Cu*c_control + Cp*(c_pos1+c_pos2) + Cv*c_vel + Cen*c_en


def pendulum3_swingup_final_cost(x, goal=[-1, 0, 0], Cp=1000.0, Cv=10.0,
                                 Cen=0.0, m=0.57288, l=0.5, b=0.15,
                                 cf=0.0, g=9.81):
    # pendulum state x = [cos(theta), sin(theta), thetadot]
    eps = 1e-6
    c_pos1 = (x[0] - goal[0] + eps)**2.0
    c_pos2 = (x[1] - goal[1] + eps)**2.0
    c_vel = (x[2] - goal[2] + eps)**2.0
    en_g = 0.5*m*(l*goal[2])**2.0 + m*g*l*(1.0-goal[0])
    en = 0.5*m*(l*x[2])**2.0 + m*g*l*(1.0-x[0])
    c_en = (en-en_g+eps)**2.0
    return Cp*(c_pos1+c_pos2) + Cv*c_vel + Cen*c_en
