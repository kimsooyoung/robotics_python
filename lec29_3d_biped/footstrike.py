import numpy as np
# from humanoid_rhs import humanoid_rhs
from cython_dynamics import humanoid_rhs_cython
from footimpulse import foot_impulse


def footstrike(t, z0, params):

    x, xd, y, yd, z, zd, \
        phi, phid, theta, thetad, psi, psid, \
        phi_lh, phi_lhd, theta_lh, theta_lhd, \
        psi_lh, psi_lhd, theta_lk, theta_lkd, \
        phi_rh, phi_rhd, theta_rh, theta_rhd, \
        psi_rh, psi_rhd, theta_rk, theta_rkd = z0

    P = params.P

    mb, mt, mc = params.mb, params.mt, params.mc
    Ibx, Iby, Ibz = params.Ibx, params.Iby, params.Ibz
    Itx, Ity, Itz = params.Itx, params.Ity, params.Itz
    Icx, Icy, Icz = params.Icx, params.Icy, params.Icz
    l0, l1, l2 = params.l0, params.l1, params.l2
    w, g = params.w, params.g

    params_arr = np.array([
        mb, mt, mc,
        Ibx, Iby, Ibz,
        Itx, Ity, Itz,
        Icx, Icy, Icz,
        l0, l1, l2,
        w, g
    ])

    # A, _, J_l, J_r, _, _ = humanoid_rhs(z, t, params)
    A, _, J_l, J_r, _, _ = humanoid_rhs_cython.humanoid_rhs(z0, t, params_arr)

    qdot_minus = np.array([
        xd, yd, zd, phid, thetad, psid, \
        phi_lhd, theta_lhd, psi_lhd, theta_lkd, \
        phi_rhd, theta_rhd, psi_rhd, theta_rkd
    ])

    I_LA, I_RA = foot_impulse(P, l2, phi, phi_lh, phi_rh, psi_lh, psi_rh, psi, theta, theta_lh, theta_lk, theta_rh, theta_rk)

    if params.stance_foot == 'right':
        P_RA = I_RA

        A_hs = np.block([
            [A, -J_l.T],
            [J_l, np.zeros((3, 3))]
        ])

        m, n = A.shape

        b_hs = np.block([
            [np.reshape(A@qdot_minus + J_r.T@P_RA, (m, 1))],
            [np.zeros((3, 1))]
        ])

        X_hs = np.linalg.solve(A_hs, b_hs)
        P_LA = X_hs[14:17, 0]
    if params.stance_foot == 'left':
        P_LA = I_LA

        A_hs = np.block([
            [A, -J_r.T],
            [J_r, np.zeros((3, 3))]
        ])

        m, n = A.shape

        b_hs = np.block([
            [np.reshape(A@qdot_minus + J_l.T@P_LA, (m,1))],
            [np.zeros((3, 1))]
        ])

        X_hs = np.linalg.solve(A_hs, b_hs)
        P_RA = X_hs[14:17, 0]

    # print(f"X_hs: {X_hs}")

    # print(f"params.stance_foot: {params.stance_foot}")
    # print(f"I_LA: {I_LA}")
    # print(f"I_RA: {I_RA}")
    # print(f"P_LA: {P_LA}")
    # print(f"P_RA: {P_RA}")

    xd, yd, zd = X_hs[0:3, 0]
    phid, thetad, psid = X_hs[3:6, 0]
    phi_lhd, theta_lhd, psi_lhd = X_hs[6:9, 0]
    theta_lkd, phi_rhd, theta_rhd = X_hs[9:12, 0]
    psi_rhd, theta_rkd = X_hs[12:14, 0]

    z_plus = np.array([
        x, xd, y, yd, z, zd, phi, phid, theta, thetad, psi, psid,\
        phi_lh, phi_lhd, theta_lh, theta_lhd,\
        psi_lh, psi_lhd, theta_lk, theta_lkd,\
        phi_rh, phi_rhd, theta_rh, theta_rhd,\
        psi_rh, psi_rhd, theta_rk, theta_rkd
    ])

    return z_plus, P_LA, P_RA
