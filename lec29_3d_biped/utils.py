import numpy as np
from copy import deepcopy
from one_step import one_step
from scipy.optimize import fsolve

def fixedpt(z0, params):
    
    params.stance_foot = params.stance_foot_init
    
    # z_temp = np.hstack(( np.zeros(6), z0 ))
    
    # 22
    phi, phid, theta, thetad, psi, psid, \
        phi_lh, phi_lhd, theta_lh, theta_lhd, \
        psi_lh, psi_lhd, theta_lk, theta_lkd, \
        phi_rh, phi_rhd, theta_rh, theta_rhd, \
        psi_rh, psi_rhd, theta_rk, theta_rkd = z0

    

    z_aft = one_step(z0, params, 1)
    
    # 22
    phi_f, phid_f, theta_f, thetad_f, psi_f, psid_f, \
        phi_lh_f, phi_lhd_f, theta_lh_f, theta_lhd_f, \
        psi_lh_f, psi_lhd_f, theta_lk_f, theta_lkd_f, \
        phi_rh_f, phi_rhd_f, theta_rh_f, theta_rhd_f, \
        psi_rh_f, psi_rhd_f, theta_rk_f, theta_rkd_f = z_aft
    
    print(f"z0: {z0}")
    print(f"z_aft: {z_aft}")
    
    return [
        phi - phi_f, phid - phid_f, 
        theta - theta_f, thetad - thetad_f, 
        psi - psi_f, psid - psid_f, \
        phi_lh - phi_lh_f, phi_lhd - phi_lhd_f, \
        theta_lh - theta_lh_f, theta_lhd - theta_lhd_f, \
        psi_lh - psi_lh_f, psi_lhd - psi_lhd_f, \
        theta_lk - theta_lk_f, theta_lkd - theta_lkd_f, \
        phi_rh - phi_rh_f, phi_rhd - phi_rhd_f, \
        theta_rh - theta_rh_f, theta_rhd - theta_rhd_f, \
        psi_rh - psi_rh_f, psi_rhd - psi_rhd_f, \
        theta_rk - theta_rk_f, theta_rkd - theta_rkd_f
    ]

def partial_jacobian(z, params):

    m = len(z)
    J = np.zeros((m, m))

    epsilon = 1e-5

    for i in range(m):
        # LIST IS IMMUATABLE 
        z_minus = deepcopy(z)
        z_plus  = deepcopy(z)

        z_minus[i] = z[i] - epsilon
        z_plus[i]  = z[i] + epsilon

        z_minus_result, _ = one_step(z_minus, 0, params, False)
        z_plus_result, _  = one_step(z_plus, 0, params, False)

        for j in range(m):
            J[j, i] = (z_plus_result[-1,j] - z_minus_result[-1,j]) / (2 * epsilon)

    return J

def find_fixed_points(z0, params):
    
    print("\n1) Root finding \n")
    
    z_star = fsolve(fixedpt, z0, params, xtol=1e-8)
    
    print(f"Fixed point z_star : \n{z_star}")
    J_star = partial_jacobian(z_star, params)
    eig_val, eig_vec = np.linalg.eig(J_star)
    
    print(f"EigenValues for linearized map \n{eig_val}")
    print(f"EigenVectors for linearized map \n{eig_vec}")
    print(f"max(abs(eigVal)) : {max(np.abs(eig_val))}")
    
    # 22
    return z_star