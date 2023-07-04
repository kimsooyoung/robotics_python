import numpy as np

from traj import traj
# from humanoid_rhs import humanoid_rhs
from cython_dynamics import humanoid_rhs_cython

def controller(z0, t, params):
    
    x, xd, y, yd, z, zd, phi, phid, theta, thetad, psi, psid, \
        phi_lh, phi_lhd, theta_lh, theta_lhd, \
        psi_lh, psi_lhd, theta_lk, theta_lkd, \
        phi_rh, phi_rhd, theta_rh, theta_rhd, \
        psi_rh, psi_rhd, theta_rk, theta_rkd = z0
        
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
    
    t0 = params.t0
    tf = params.tf
    s0 = params.s0
    sf = params.sf
    v0 = params.v0
    vf = params.vf
    a0 = params.a0
    af = params.af
    
    Xdd_des = np.zeros((8,1))
    Xd_des = np.zeros((8,1))
    X_des = np.zeros((8,1))
    
    for i in range(8):
        Xdd_des[i], Xd_des[i], X_des[i] = traj(
            t, t0, tf, 
            s0[i], sf[i], 
            v0[i], vf[i], 
            a0[i], af[i]
        )
    
    B = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0], # 3 for linear pos global
        
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0], # 3 for ang pos global
        
        [1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1], # 8 controlllable part 
        
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0], # 3 for P's
    ])
    
    S_L = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # phi
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # theta
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # psi
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # phi_rh
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # theta_rh
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # psi_rh
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # theta_lh
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # theta_rk
    ])

    S_R = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #phi
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #theta
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #psi
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #phi_lh
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #theta_lh
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #psi_lh
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #theta_lh
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] #theta_rk;
    ])
    
    M, N, J_l, J_r, Jdot_l, Jdot_r = humanoid_rhs_cython.humanoid_rhs(z0, t, params_arr)
    
    qdot = np.array([
        xd, yd, zd, phid, thetad, psid,
        phi_lhd, theta_lhd, psi_lhd, theta_lkd,
        phi_rhd, theta_rhd, psi_rhd, theta_rkd
    ])
    
    ### AX = b (without control) ###
    if params.stance_foot == 'right':
        AA = np.block([
            [M, -J_r.T], 
            [J_r, np.zeros((3,3))]
        ])
        
        bb = np.block([
            [N],
            [ np.reshape(-Jdot_r @ qdot.T, (3, 1)) ]
        ])
    elif params.stance_foot == 'left':
        AA = np.block([
            [M, -J_l.T],
            [J_l, np.zeros((3,3))]
        ])
        
        bb = np.block([
            [N],
            [ np.reshape(-Jdot_l @ qdot.T, (3, 1)) ]
        ])

    ########### AX = b + B*tau  (with control) #######
    ### Reduced X_c = S X = S*inv(A) (b+B*tau) = v ###
    ### Solving for tau gives,                     ###
    ### tau = inv(S*inv(A)*B)*(v - S*inv(A)*b)     ###
    ##################################################
    Kp = params.Kp
    Kd = params.Kd
    Ainv = np.linalg.inv(AA)
    
    if params.stance_foot == 'right':
        X = np.array([
            [phi, theta, psi, phi_lh, theta_lh, psi_lh, theta_lk, theta_rk]
        ]).T
        
        Xd = np.array([
            [phid, thetad, psid, phi_lhd, theta_lhd, psi_lhd, theta_lkd, theta_rkd]
        ]).T
        
        v = Xdd_des + Kd*(Xd_des-Xd) + Kp*(X_des-X)
        # print(f"v: {v}")
        # print(f"Xd_des-Xd: {Xd_des-Xd}")
        
        SAinvB = S_R @ Ainv @ B
        SAinvB_inv = np.linalg.inv(SAinvB)
        tau = SAinvB_inv @ (v - S_R @ Ainv @ bb)
    elif params.stance_foot == 'left':
        X = np.array([
            [phi, theta, psi, phi_rh, theta_rh, psi_rh, theta_lk, theta_rk]
        ]).T
        
        Xd = np.array([
            [phid, thetad, psid, phi_rhd, theta_rhd, psi_rhd, theta_lkd, theta_rkd]
        ]).T
        
        v = Xdd_des + Kd*(Xd_des-Xd) + Kp*(X_des-X)
        # print(f"v: {v}")
        # print(f"Xd_des-Xd: {Xd_des-Xd}")
        
        SAinvB = S_L @ Ainv @ B
        SAinvB_inv = np.linalg.inv(SAinvB)
        tau = SAinvB_inv @ (v - S_L @ Ainv @ bb)
    
    return tau