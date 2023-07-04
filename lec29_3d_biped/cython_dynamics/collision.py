import numpy as np

def sin(theta):
    return np.sin(theta)

def cos(theta):
    return np.cos(theta)

def collision(t, z_in, params):
    
    x, xd, y, yd, z, zd, \
        phi, phid, theta, thetad, psi, psid, \
        phi_lh, phi_lhd, theta_lh, theta_lhd, \
        psi_lh, psi_lhd, theta_lk, theta_lkd, \
        phi_rh, phi_rhd, theta_rh, theta_rhd, \
        psi_rh, psi_rhd, theta_rk, theta_rkd = z_in
        
    l1, l2, w = params
    
    gstop = 2*w*cos(psi)*sin(phi) + 2*w*cos(phi)*sin(psi)*sin(theta) - l1*cos(phi)*cos(phi_lh)*cos(theta)*cos(theta_lh) + l1*cos(phi)*cos(phi_rh)*cos(theta)*cos(theta_rh) + l1*cos(psi_lh)*sin(phi)*sin(psi)*sin(theta_lh) + l1*cos(psi)*sin(phi)*sin(psi_lh)*sin(theta_lh) - l1*cos(psi_rh)*sin(phi)*sin(psi)*sin(theta_rh) + l1*cos(psi)*sin(phi)*sin(psi_rh)*sin(theta_rh) + l1*cos(psi_lh)*cos(psi)*cos(theta_lh)*sin(phi)*sin(phi_lh) + l1*cos(psi_rh)*cos(psi)*cos(theta_rh)*sin(phi)*sin(phi_rh) - l1*cos(phi)*cos(psi_lh)*cos(psi)*sin(theta)*sin(theta_lh) + l1*cos(phi)*cos(psi_rh)*cos(psi)*sin(theta)*sin(theta_rh) + l2*cos(phi)*cos(phi_lh)*cos(theta)*sin(theta_lh)*sin(theta_lk) - l2*cos(phi)*cos(phi_rh)*cos(theta)*sin(theta_rh)*sin(theta_rk) + l2*cos(psi_lh)*cos(theta_lh)*sin(phi)*sin(psi)*sin(theta_lk) + l2*cos(psi_lh)*cos(theta_lk)*sin(phi)*sin(psi)*sin(theta_lh) + l2*cos(psi)*cos(theta_lh)*sin(phi)*sin(psi_lh)*sin(theta_lk) + l2*cos(psi)*cos(theta_lk)*sin(phi)*sin(psi_lh)*sin(theta_lh) - l2*cos(psi_rh)*cos(theta_rh)*sin(phi)*sin(psi)*sin(theta_rk) - l2*cos(psi_rh)*cos(theta_rk)*sin(phi)*sin(psi)*sin(theta_rh) + l2*cos(psi)*cos(theta_rh)*sin(phi)*sin(psi_rh)*sin(theta_rk) + l2*cos(psi)*cos(theta_rk)*sin(phi)*sin(psi_rh)*sin(theta_rh) - l1*cos(theta_lh)*sin(phi)*sin(phi_lh)*sin(psi_lh)*sin(psi) + l1*cos(theta_rh)*sin(phi)*sin(phi_rh)*sin(psi_rh)*sin(psi) + l1*cos(phi)*sin(psi_lh)*sin(psi)*sin(theta)*sin(theta_lh) + l1*cos(phi)*sin(psi_rh)*sin(psi)*sin(theta)*sin(theta_rh) - l2*cos(phi)*cos(phi_lh)*cos(theta)*cos(theta_lh)*cos(theta_lk) + l2*cos(phi)*cos(phi_rh)*cos(theta)*cos(theta_rh)*cos(theta_rk) - l2*cos(psi_lh)*cos(psi)*sin(phi)*sin(phi_lh)*sin(theta_lh)*sin(theta_lk) - l2*cos(theta_lh)*cos(theta_lk)*sin(phi)*sin(phi_lh)*sin(psi_lh)*sin(psi) - l2*cos(psi_rh)*cos(psi)*sin(phi)*sin(phi_rh)*sin(theta_rh)*sin(theta_rk) + l2*cos(theta_rh)*cos(theta_rk)*sin(phi)*sin(phi_rh)*sin(psi_rh)*sin(psi) + l2*cos(phi)*cos(theta_lh)*sin(psi_lh)*sin(psi)*sin(theta)*sin(theta_lk) + l2*cos(phi)*cos(theta_lk)*sin(psi_lh)*sin(psi)*sin(theta)*sin(theta_lh) + l2*cos(phi)*cos(theta_rh)*sin(psi_rh)*sin(psi)*sin(theta)*sin(theta_rk) + l2*cos(phi)*cos(theta_rk)*sin(psi_rh)*sin(psi)*sin(theta)*sin(theta_rh) + l2*sin(phi)*sin(phi_lh)*sin(psi_lh)*sin(psi)*sin(theta_lh)*sin(theta_lk) - l2*sin(phi)*sin(phi_rh)*sin(psi_rh)*sin(psi)*sin(theta_rh)*sin(theta_rk) + l2*cos(psi_lh)*cos(psi)*cos(theta_lh)*cos(theta_lk)*sin(phi)*sin(phi_lh) + l2*cos(psi_rh)*cos(psi)*cos(theta_rh)*cos(theta_rk)*sin(phi)*sin(phi_rh) - l2*cos(phi)*cos(psi_lh)*cos(psi)*cos(theta_lh)*sin(theta)*sin(theta_lk) - l2*cos(phi)*cos(psi_lh)*cos(psi)*cos(theta_lk)*sin(theta)*sin(theta_lh) + l2*cos(phi)*cos(psi_rh)*cos(psi)*cos(theta_rh)*sin(theta)*sin(theta_rk) + l2*cos(phi)*cos(psi_rh)*cos(psi)*cos(theta_rk)*sin(theta)*sin(theta_rh) + l1*cos(phi)*cos(psi_lh)*cos(theta_lh)*sin(phi_lh)*sin(psi)*sin(theta) + l1*cos(phi)*cos(psi)*cos(theta_lh)*sin(phi_lh)*sin(psi_lh)*sin(theta) + l1*cos(phi)*cos(psi_rh)*cos(theta_rh)*sin(phi_rh)*sin(psi)*sin(theta) - l1*cos(phi)*cos(psi)*cos(theta_rh)*sin(phi_rh)*sin(psi_rh)*sin(theta) + l2*cos(phi)*cos(psi_lh)*cos(theta_lh)*cos(theta_lk)*sin(phi_lh)*sin(psi)*sin(theta) + l2*cos(phi)*cos(psi)*cos(theta_lh)*cos(theta_lk)*sin(phi_lh)*sin(psi_lh)*sin(theta) + l2*cos(phi)*cos(psi_rh)*cos(theta_rh)*cos(theta_rk)*sin(phi_rh)*sin(psi)*sin(theta) - l2*cos(phi)*cos(psi)*cos(theta_rh)*cos(theta_rk)*sin(phi_rh)*sin(psi_rh)*sin(theta) - l2*cos(phi)*cos(psi_lh)*sin(phi_lh)*sin(psi)*sin(theta)*sin(theta_lh)*sin(theta_lk) - l2*cos(phi)*cos(psi)*sin(phi_lh)*sin(psi_lh)*sin(theta)*sin(theta_lh)*sin(theta_lk) - l2*cos(phi)*cos(psi_rh)*sin(phi_rh)*sin(psi)*sin(theta)*sin(theta_rh)*sin(theta_rk) + l2*cos(phi)*cos(psi)*sin(phi_rh)*sin(psi_rh)*sin(theta)*sin(theta_rh)*sin(theta_rk)
    
    return gstop
