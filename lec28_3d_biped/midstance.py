
def midstance(t, z, params):
    
    l0, l1, l2 = params.l0, params.l1, params.l2
    w = params.w
    
    x, xd, y, yd, z, zd, \ 
        phi, phid, theta, thetad, psi, psid, \ 
        phi_lh, phi_lhd, theta_lh, theta_lhd, \ 
        psi_lh, psi_lhd, theta_lk, theta_lkd, \ 
        phi_rh, phi_rhd, theta_rh, theta_rhd, \ 
        psi_rh, psi_rhd, theta_rk, theta_rkd = z #28
        
    g_stop = zd # l1*cos(theta0 + theta1) - l1*cos(theta0 + theta2) + l2*cos(theta0 + theta1 + theta3) - l2*cos(theta0 + theta2 + theta4)   

    return g_stop
