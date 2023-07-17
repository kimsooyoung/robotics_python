import time
import numpy as np 
from controller import controller
# from humanoid_rhs import humanoid_rhs
from cython_dynamics import humanoid_rhs_cython

def single_stance_helper(B, z0, t, params):
    
    # start = time.time()

    tau = controller(z0, t, params)
    
    # end = time.time()
    # print(f"mid1 time : {end - start:.5f} sec")

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
    
    # print(f"params_arr: {params_arr}")
    
    A, b, J_l, J_r, Jdot_l, Jdot_r = humanoid_rhs_cython.humanoid_rhs(z0, t, params_arr) 
    
    # end = time.time()
    # print(f"mid2 time : {end - start:.5f} sec")
    
    # P: impact force
    x, P_RA, P_LA = None, None, None
    qdot = np.array([
        xd, yd, zd, phid, thetad, psid,
        phi_lhd, theta_lhd, psi_lhd, theta_lkd,
        phi_rhd, theta_rhd, psi_rhd, theta_rkd
    ])
    
    if params.stance_foot == 'right':
        AA = np.block([
            [A, -J_r.T], 
            [J_r, np.zeros((3,3))]
        ])
        
        bb = np.block([
            [b],
            [ np.reshape(-Jdot_r @ qdot.T, (3, 1)) ]
        ]) + B @ tau
        
        # x = np.linalg.solve(AA, bb)
        AA_inv = np.linalg.inv(AA)
        x = AA_inv @ bb
        
        P_RA = np.array([ x[14,0], x[15,0], x[16,0] ])
        P_LA = np.array([ 0, 0, 0 ])
    elif params.stance_foot == 'left':
        AA = np.block([
            [A, -J_l.T],
            [J_l, np.zeros((3,3))]
        ])
        
        bb = np.block([
            [b],
            [ np.reshape(-Jdot_l @ qdot.T, (3, 1)) ]
        ]) + B @ tau
        
        # x = np.linalg.solve(AA, bb)
        AA_inv = np.linalg.inv(AA)
        x = AA_inv @ bb
        
        P_LA = np.array([ x[14,0], x[15,0], x[16,0] ])
        P_RA = np.array([ 0, 0, 0 ])

    # end = time.time()
    # print(f"mid3 time : {end - start:.5f} sec")


    return x, A, b, P_RA, P_LA, tau

def single_stance(t, z, params):
    
    _, xd, _, yd, _, zd, _, phid, _, thetad, _, psid, \
        phi_lh, phi_lhd, theta_lh, theta_lhd, \
        psi_lh, psi_lhd, theta_lk, theta_lkd, \
        phi_rh, phi_rhd, theta_rh, theta_rhd, \
        psi_rh, psi_rhd, theta_rk, theta_rkd = z

    B = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    
    # end = time.time()
    # print(f"mid1 time : {end - start:.5f} sec")

    x, A, b, P_RA, P_LA, tau = single_stance_helper(B, z, t, params)
    
    xdd = x[0,0]; ydd = x[1,0]; zdd = x[2,0]; 
    phidd = x[3,0]; thetadd = x[4,0]; psidd = x[5,0];
    
    phi_lhdd = x[6,0]; theta_lhdd = x[7,0]; 
    psi_lhdd = x[8,0]; theta_lkdd = x[9,0];
    
    phi_rhdd = x[10,0]; theta_rhdd = x[11,0];
    psi_rhdd = x[12,0]; theta_rkdd = x[13,0];
    
    # if t == 0.0:
    #     print(f"x: {x}")
    
    zdot = np.array([
        xd, xdd, yd, ydd, zd, zdd, phid, phidd, thetad, thetadd, psid, psidd, \
        phi_lhd, phi_lhdd, theta_lhd, theta_lhdd, \
        psi_lhd, psi_lhdd, theta_lkd, theta_lkdd, \
        phi_rhd, phi_rhdd, theta_rhd, theta_rhdd, \
        psi_rhd, psi_rhdd, theta_rkd, theta_rkdd
    ])

    # print(f"single stance zdot: {zdot}")
    
    return zdot