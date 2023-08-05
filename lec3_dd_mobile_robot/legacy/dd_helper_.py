import math

import numpy as np
from scipy import interpolate


def euler_integration(tspan, z0, u):
    v = u[0]
    omega = u[1]
    h = tspan[1] - tspan[0]

    x0 = z0[0]
    y0 = z0[1]
    theta0 = z0[2]

    xdot_c = v * math.cos(theta0)
    ydot_c = v * math.sin(theta0)
    thetadot = omega

    x1 = x0 + xdot_c * h
    y1 = y0 + ydot_c * h
    theta1 = theta0 + thetadot * h

    z1 = [x1, y1, theta1]
    return z1


# offset이 적용되는 함수
def ptP_to_ptC(state, params):
    # x_p, y_p : 로봇이 도달해야 할 좌표
    # params.px,params.py : 로봇이 회전 후 가져야 할 offset
    # 실제 해당 offset 만큼은 이동을 해야 하므로 -np.matmul이 되었다.
    x_p, y_p, theta = state

    cos = np.cos(theta)
    sin = np.sin(theta)
    R = np.array([[cos, -sin],
                  [sin, cos]])
    r = np.array([params.px, params.py])
    p = np.array([x_p, y_p])
    c = -np.matmul(R, np.transpose(r)) + np.transpose(p)

    return c


# ptP_to_ptC를 복원하는 함수
def ptC_to_ptP(state, params):
    x_c, y_c, theta = state

    cos = np.cos(theta)
    sin = np.sin(theta)

    R = np.array([[cos, -sin],
                  [sin, cos]])
    r = np.array([params.px, params.py])
    c = np.array([x_c, y_c])
    p = np.matmul(R, np.transpose(r)) + np.transpose(c)

    return p


def interpolation(params, z, traj):

    # interpolation
    t = np.arange(0, params.t_length, 0.01)
    t_interp = np.arange(0, params.t_length, 1/params.fps)
    f_z1 = interpolate.interp1d(t, z[:, 0])
    f_z2 = interpolate.interp1d(t, z[:, 1])
    f_z3 = interpolate.interp1d(t, z[:, 2])

    shape = (len(t_interp), 3)
    z_interp = np.zeros(shape)
    z_interp[:, 0] = f_z1(t_interp)
    z_interp[:, 1] = f_z2(t_interp)
    z_interp[:, 2] = f_z3(t_interp)

    f_p1 = interpolate.interp1d(t, traj[:, 0])
    f_p2 = interpolate.interp1d(t, traj[:, 1])
    shape = (len(t_interp), 2)
    p_interp = np.zeros(shape)
    p_interp[:, 0] = f_p1(t_interp)
    p_interp[:, 1] = f_p2(t_interp)

    return t_interp, z_interp, p_interp
