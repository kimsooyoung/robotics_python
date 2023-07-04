import numpy as np
from .collision_cython import collision as collision_cython

def collision(t, z_in, params):
    
    x, xd, y, yd, z, zd, \
        phi, phid, theta, thetad, psi, psid, \
        phi_lh, phi_lhd, theta_lh, theta_lhd, \
        psi_lh, psi_lhd, theta_lk, theta_lkd, \
        phi_rh, phi_rhd, theta_rh, theta_rhd, \
        psi_rh, psi_rhd, theta_rk, theta_rkd = z_in

    z_arr = np.array([
        x, xd, y, yd, z, zd,
        phi, phid, theta, thetad, psi, psid,
        phi_lh, phi_lhd, theta_lh, theta_lhd,
        psi_lh, psi_lhd, theta_lk, theta_lkd,
        phi_rh, phi_rhd, theta_rh, theta_rhd,
        psi_rh, psi_rhd, theta_rk, theta_rkd
    ])

    params_arr = np.array([
        params.l1, params.l2, params.w
    ])
    
    gstop = collision_cython(t, z_arr, params_arr)
    
    return gstop
