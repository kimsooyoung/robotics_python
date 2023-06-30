import numpy as np 
from controller import controller
from humanoid_rhs import humanoid_rhs

def single_stance_helper(B, z, t, params):
    
    tau = controller(z, t, params)
    
    x, xd, y, yd, z, zd, phi, phid, theta, thetad, psi, psid, \
        phi_lh, phi_lhd, theta_lh, theta_lhd, \
        psi_lh, psi_lhd, theta_lk, theta_lkd, \
        phi_rh, phi_rhd, theta_rh, theta_rhd, \
        psi_rh, psi_rhd, theta_rk, theta_rkd = z

    mb, mt, mc = params.mb, params.mt, params.mc
    Ibx, Iby, Ibz = params.Ibx, params.Iby, params.Ibz
    Itx, Ity, Itz = params.Itx, params.Ity, params.Itz
    Icx, Icy, Icz = params.Icx, params.Icy, params.Icz
    l0, l1, l2 = params.l0, params.l1, params.l2
    w, g = params.w, params.g

    tau = controller(t,zz,params)
    
    A, b, J_l, J_r, Jdot_l, Jdot_r = humanoid_rhs(t, z, params) 
    
    qdot = np.array([xd yd zd phid thetad psid phi_lhd theta_lhd psi_lhd theta_lkd phi_rhd theta_rhd psi_rhd theta_rkd]) 
    
    # P: impact force
    x, P_RA, P_LA = None
    if params.stance_foot == 'right':
        AA = np.block([
            [A, -J_r.T], 
            [J_r, np.zeros((3,3))]
        ])
        
        bb = np.block([
            [b],
            [ np.reshape(-Jdot_r @ qdot.T, (3, 1)) ]
        ]) + B @ tau
        
        x = np.linalg.solve(AA, bb)
        P_RA = np.array([ x[14], x[15], x[16] ])
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
        
        x = np.linalg.solve(AA, bb)
        P_LA = np.array([ x[14], x[15], x[16] ])
        P_RA = np.array([ 0, 0, 0 ])
        
    return x, A, b, P_RA, P_LA, tau

def single_stance(z, t, params):

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
    
    x, A, b, P_RA, P_LA, tau = single_stance_helper(B, z, t, params)
    
    xdd = x[0]; ydd = x[1]; zdd = x[2]; 
    phidd = x[3]; thetadd = x[4]; psidd = x[5];
    
    phi_lhdd = x[6]; theta_lhdd = x[7]; 
    psi_lhdd = x[8]; theta_lkdd = x[9];
    
    phi_rhdd = x[10]; theta_rhdd = x[11];
    psi_rhdd = x[12]; theta_rkdd = x[13];
    
    zdot = np.array([
        xd, xdd, yd, ydd, zd, zdd, phid, phidd, thetad, thetadd, psid, psidd, \
        phi_lhd, phi_lhdd, theta_lhd, theta_lhdd, \
        psi_lhd, psi_lhdd, theta_lkd, theta_lkdd, \
        phi_rhd, phi_rhdd, theta_rhd, theta_rhdd, \
        psi_rhd, psi_rhdd, theta_rkd, theta_rkdd
    ])
    
    return zdot