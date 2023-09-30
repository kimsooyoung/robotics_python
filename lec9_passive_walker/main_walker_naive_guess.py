# Copyright 2022 @RoadBalance
# Reference from https://pab47.github.io/legs.html
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import interpolate
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


class parameters:

    def __init__(self):

        self.g = 1
        self.m = 0.5
        self.M = 1
        self.I = 0.02
        self.l = 1.0
        self.c = 0.5
        self.gam = 0.01
        self.pause = 0.02
        self.fps = 10


def sin(angle):
    return np.sin(angle)


def cos(angle):
    return np.cos(angle)


def animate(t, z, parms):
    # interpolation
    data_pts = 1/parms.fps
    t_interp = np.arange(t[0], t[len(t)-1], data_pts)
    [m, n] = np.shape(z)
    shape = (len(t_interp), n)
    z_interp = np.zeros(shape)

    for i in range(0, n):
        f = interpolate.interp1d(t, z[:, i])
        z_interp[:, i] = f(t_interp)

    l = parms.l
    c = parms.c

    min_xh = min(z[:, 4])
    max_xh = max(z[:, 4])
    dist_travelled = max_xh - min_xh
    camera_rate = dist_travelled/len(t_interp)

    window_xmin = -1*l
    window_xmax = 1*l
    window_ymin = -0.1
    window_ymax = 1.1*l

    R1 = np.array([min_xh - l, 0])
    R2 = np.array([max_xh + l, 0])

    ramp, = plt.plot([R1[0], R2[0]], [R1[1], R2[1]], linewidth=5, color='black')

    # plot
    for i in range(0, len(t_interp)):
        theta1 = z_interp[i, 0]
        theta2 = z_interp[i, 2]
        xh = z_interp[i, 4]
        yh = z_interp[i, 5]

        H = np.array([xh, yh])
        C1 = np.array([xh+l*sin(theta1), yh-l*cos(theta1)])
        G1 = np.array([xh+c*sin(theta1), yh-c*cos(theta1)])
        C2 = np.array([xh+l*sin(theta1+theta2), yh-l*cos(theta1+theta2)])
        G2 = np.array([xh+c*sin(theta1+theta2), yh-c*cos(theta1+theta2)])

        leg1, = plt.plot([H[0], C1[0]], [H[1], C1[1]], linewidth=5, color='red')
        leg2, = plt.plot([H[0], C2[0]], [H[1], C2[1]], linewidth=5, color='red')
        com1, = plt.plot(G1[0], G1[1], color='black', marker='o', markersize=5)
        com2, = plt.plot(G2[0], G2[1], color='black', marker='o', markersize=5)
        hip, = plt.plot(H[0], H[1], color='black', marker='o', markersize=10)

        window_xmin = window_xmin + camera_rate
        window_xmax = window_xmax + camera_rate
        plt.xlim(window_xmin, window_xmax)
        plt.ylim(window_ymin, window_ymax)
        plt.gca().set_aspect('equal')

        plt.pause(parms.pause)
        hip.remove()
        leg1.remove()
        leg2.remove()
        com1.remove()
        com2.remove()

    plt.close()


def n_steps(t0, z0, parms, steps):

    z = z0
    t = t0

    theta1 = z0[0]
    
    l = parms.l

    xh = 0
    yh = l*cos(theta1)
    xh_start = xh
    zz = np.append(z0,np.array([xh, yh]))

    for i in range(0, steps):
        [z_temp, t_temp] = one_step(t0, z0, parms)
        [mm, nn] = np.shape(z_temp)
        zz_temp = np.zeros((mm, 6))
        for j in range(0, mm):
            xh = xh_start + l*sin(z_temp[0, 0]) - l*sin(z_temp[j, 0])
            yh = l*cos(z_temp[j, 0])
            zz_temp[j, :] = np.append(z_temp[j,:],np.array([xh, yh]))

        xh_start = zz_temp[mm-2, 4]

        if i == 0:
            # z = np.concatenate(([z], z_temp[1:mm-1,:]), axis=0)
            t = np.concatenate(([t], t_temp[1:mm-1]), axis=0)
            zz = np.concatenate(([zz], zz_temp[1:mm-1, :]), axis=0)
        else:
            # z = np.concatenate((z, z_temp[1:mm-1,:]), axis=0)
            t = np.concatenate((t, t_temp[1:mm-1]), axis=0)
            zz = np.concatenate((zz, zz_temp[1:mm-1, :]), axis=0)

        theta1 = z_temp[mm-1, 0]
        omega1 = z_temp[mm-1, 1]
        theta2 = z_temp[mm-1, 2]
        omega2 = z_temp[mm-1, 3]

        z0 = np.array([theta1, omega1, theta2, omega2])
        t0 = t_temp[mm-1]

    return zz, t


def one_step(t0, z0, parms):

    tf = t0 + 4
    t = np.linspace(t0, tf, 1001)
    collision.terminal = True
    # contact.direction = -1

    # error correction을 위해 RK45를 사용
    sol = solve_ivp(
        single_stance, [t0, tf], z0, method='RK45', t_eval=t,
        dense_output=True, events=collision, atol=1e-13, rtol=1e-12,
        args=(parms.M,parms.m,parms.I,parms.l,parms.c,parms.g,parms.gam)
    )

    # get solution at different time steps from sol.y
    [m, n] = np.shape(sol.y)
    shape = (n, m)
    t = sol.t
    z = np.zeros(shape)

    # get event from sol.y_events and exact time sol.t_events
    [mm, nn, pp] = np.shape(sol.y_events)
    tt_last_event = sol.t_events[mm-1]
    yy_last_event = sol.y_events[mm-1]

    # save data in z
    for i in range(0, m):
        z[:, i] = sol.y[i, :]

    #get state before footstrike using events
    t_end = tt_last_event[0]
    theta1 = yy_last_event[0, 0]
    omega1 = yy_last_event[0, 1]
    theta2 = yy_last_event[0, 2]
    omega2 = yy_last_event[0, 3]

    zminus = np.array([theta1, omega1, theta2, omega2])

    # return state after footstrike
    zplus = footstrike(t_end, zminus, parms)

    # replace last entry in z and t
    t[n-1] = t_end
    z[n-1, 0] = zplus[0]
    z[n-1, 1] = zplus[1]
    z[n-1, 2] = zplus[2]
    z[n-1, 3] = zplus[3]

    return z, t


def collision(t, z, M, m, I, l, c, g, gam):

    theta1, omega1, theta2, omega2 = z

    # allow legs to pass through for small hip angles
    # (taken care in real walker using stepping stones)
    if (theta1 > -0.05):
        gstop = 1
    else:
        gstop = theta2 + 2*theta1

    return gstop


def footstrike(t, z, parms):

    theta1_n,omega1_n,theta2_n,omega2_n = z

    theta1 = theta1_n + theta2_n;
    theta2 = -theta2_n;

    M = parms.M
    m = parms.m
    I = parms.I
    l = parms.l
    c = parms.c
    g = parms.g
    gam = parms.gam

    J11 =  1
    J12 =  0
    J13 =  l*(-cos(theta1_n) + cos(theta1_n + theta2_n))
    J14 =  l*cos(theta1_n + theta2_n)
    J21 =  0
    J22 =  1
    J23 =  l*(-sin(theta1_n) + sin(theta1_n + theta2_n))
    J24 =  l*sin(theta1_n + theta2_n)

    J = np.array([[J11, J12, J13, J14], [J21,J22,J23,J24]])

    A11 =  1.0*M + 2.0*m
    A12 =  0
    A13 =  -1.0*M*l*cos(theta1_n) + m*(c - l)*cos(theta1_n) + 1.0*m*(c*cos(theta1_n + theta2_n) - l*cos(theta1_n))
    A14 =  1.0*c*m*cos(theta1_n + theta2_n)
    A21 =  0
    A22 =  1.0*M + 2.0*m
    A23 =  -1.0*M*l*sin(theta1_n) + m*(c - l)*sin(theta1_n) + m*(c*sin(theta1_n + theta2_n) - l*sin(theta1_n))
    A24 =  1.0*c*m*sin(theta1_n + theta2_n)
    A31 =  -1.0*M*l*cos(theta1_n) + m*(c - l)*cos(theta1_n) + 1.0*m*(c*cos(theta1_n + theta2_n) - l*cos(theta1_n))
    A32 =  -1.0*M*l*sin(theta1_n) + m*(c - l)*sin(theta1_n) + m*(c*sin(theta1_n + theta2_n) - l*sin(theta1_n))
    A33 =  2.0*I + M*l**2 + m*(c - l)**2 + m*(c**2 - 2*c*l*cos(theta2_n) + l**2)
    A34 =  1.0*I + c*m*(c - l*cos(theta2_n))
    A41 =  1.0*c*m*cos(theta1_n + theta2_n)
    A42 =  1.0*c*m*sin(theta1_n + theta2_n)
    A43 =  1.0*I + c*m*(c - l*cos(theta2_n))
    A44 =  1.0*I + c**2*m
    A_n_hs = np.array([[A11, A12, A13, A14], [A21, A22, A23, A24], [A31, A32, A33, A34], [A41, A42, A43, A44]])

    X_n_hs = np.array([0, 0, omega1_n, omega2_n])
    b_temp  = A_n_hs.dot(X_n_hs)
    b_hs = np.block([ b_temp, 0, 0 ])
    zeros_22 = np.zeros((2,2))
    A_hs = np.block([[A_n_hs, -np.transpose(J)] , [ J, zeros_22] ])
    invA_hs = np.linalg.inv(A_hs)
    X_hs = invA_hs.dot(b_hs)
    omega1 = X_hs[2] + X_hs[3]
    omega2 = -X_hs[3]

    return [theta1,omega1,theta2,omega2]

def single_stance(t, z, M,m,I,l,c,g,gam):

    theta1,omega1,theta2,omega2 = z

    A11 =  2.0*I + M*l**2 + m*(c - l)**2 + m*(c**2 - 2*c*l*cos(theta2) + l**2)
    A12 =  1.0*I + c*m*(c - l*cos(theta2))
    A21 =  1.0*I + c*m*(c - l*cos(theta2))
    A22 =  1.0*I + c**2*m

    b1 =  -M*g*l*sin(gam - theta1) + c*g*m*sin(gam - theta1) - c*g*m*sin(-gam + theta1 + theta2) - 2*c*l*m*omega1*omega2*sin(theta2) - c*l*m*omega2**2*sin(theta2) - 2*g*l*m*sin(gam - theta1)
    b2 =  1.0*c*m*(-g*sin(-gam + theta1 + theta2) + l*omega1**2*sin(theta2))

    A_ss = np.array([[A11, A12], [A21,A22]])
    b_ss = np.array([b1,b2])

    invA_ss = np.linalg.inv(A_ss)
    thetaddot = invA_ss.dot(b_ss)
    alpha1 = thetaddot[0]
    alpha2 = thetaddot[1]

    return [omega1,alpha1,omega2,alpha2]


parms = parameters();

# this initial condition leads to periodic walking
# q1 = 0.162597833780035;
# u1 = -0.231869638058927;
# q2 = -0.325195667560070;
# u2 = 0.037978468073736;

# an initial guess
# initial angle and speed
q1 = 0.2
u1 = -0.25
q2 = -0.4
u2 = 0.2
z0 = np.array([q1,u1,q2,u2])

t0 = 0;
steps = 4;
[z,t] = n_steps(t0,z0,parms,steps)

animate(t,z,parms)

if (1):
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(t,z[:,0],'r--')
    plt.plot(t,z[:,2],'b')
    plt.ylabel('theta')
    plt.subplot(2,1,2)
    plt.plot(t,z[:,1],'r--')
    plt.plot(t,z[:,3],'b')
    plt.ylabel('thetadot')
    plt.xlabel('time')

    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(t,z[:,4],'b')
    plt.ylabel('xh')
    plt.subplot(2,1,2)
    plt.plot(t,z[:,5],'b')
    plt.ylabel('yh')
    plt.xlabel('time')

    # plt.show()
    plt.show(block=False)
    plt.pause(3)
    plt.close()
