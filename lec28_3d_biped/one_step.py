import numpy as np
from scipy.integrate import solve_ivp

from collision import collision
from midstance import midstance
from hip_positions import hip_positions
from hip_velocities import hip_velocities
from single_stance import single_stance, single_stance_helper

def one_step(z0, params, steps):
    
    l1, l2, w = params.l1, params.l2, params.w
    
    z_temp = np.hstack((np.zeros(6), z0))
    
    x, xd, y, yd, z, zd, \
        phi, phid, theta, thetad, psi, psid, \
        phi_lh, phi_lhd, theta_lh, theta_lhd, \
        psi_lh, psi_lhd, theta_lk, theta_lkd, \
        phi_rh, phi_rhd, theta_rh, theta_rhd, \
        psi_rh, psi_rhd, theta_rk, theta_rkd = z_temp
        
    pos_hip_l_stance, pos_hip_r_stance = hip_positions(
        l1, l2, phi, 
        phi_lh, phi_rh, psi_lh, psi_rh, 
        psi, theta, 
        theta_lh, theta_lk, theta_rh, theta_rk, 
        w
    )
    
    vel_hip_l_stance, vel_hip_r_stance = hip_velocities(
        l1, l2, phi, phid,
        phi_lh, phi_rh, phi_lhd, phi_rhd,
        psid, psi_lh, psi_rh, psi, psi_lhd, psi_rhd,
        theta, thetad, theta_lh, theta_lk,
        theta_rh, theta_rk,
        theta_lhd, theta_lkd, theta_rhd, theta_rkd, 
        w
    )
    
    if params.stance_foot_init == 'right':
        x, y, z = pos_hip_r_stance[0], pos_hip_r_stance[1], pos_hip_r_stance[2]
        xd, yd, zd = vel_hip_r_stance[0], vel_hip_r_stance[1], vel_hip_r_stance[2]
    elif params.stance_foot_init == 'left':
        x, y, z = pos_hip_l_stance[0], pos_hip_l_stance[1], pos_hip_l_stance[2]
        xd, yd, zd = vel_hip_l_stance[0], vel_hip_l_stance[1], vel_hip_l_stance[2]
        
    z0 = np.hstack(([x, xd, y, yd, z, zd], z0))
    
    t_ode = np.array([0.0])
    z_ode = np.zeros((1,28)); z_ode[0] = z0
    
    tf = 0
    
    P_LA_all = np.zeros( (1,3) )
    P_RA_all = np.array( (1,3) )
    Torque = np.zeros( (1,1) )
    
    for i in range(steps):
        
        t0, t1 = 0, 2
        x, xd, y, yd, z, zd, \
            phi, phid, theta, thetad, psi, psid, \
            phi_lh, phi_lhd, theta_lh, theta_lhd, \
            psi_lh, psi_lhd, theta_lk, theta_lkd, \
            phi_rh, phi_rhd, theta_rh, theta_rhd, \
            psi_rh, psi_rhd, theta_rk, theta_rkd = z0 #28
            
        t_span = np.linspace(t0, t1)
        params.t0 = 0
        params.tf = 0.2
        
        if params.stance_foot_init == 'right':
            params.s0 = np.array([phi, theta, psi, phi_lh, theta_lh, psi_lh, theta_lk, theta_rk])
            params.v0 = np.array([phid, thetad, psid, phi_lhd, theta_lhd, psi_lhd, theta_lkd, theta_rkd])
        elif params.stance_foot_init == 'left':
            params.s0 = np.array([phi, theta, psi, phi_rh, theta_rh, psi_rh, theta_lk, theta_rk])
            params.v0 = np.array([phid, thetad, psid, phi_rhd, theta_rhd, psi_rhd, theta_lkd, theta_rkd])
        params.sf = np.array([0, 0, 0, 0, params.stepAngle, 0, 0, 0])
        params.vf = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        params.a0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        params.af = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        collision.terminal = True
        collision.direction = 0
        
        sol = solve_ivp(
            single_stance, [t0, t1], z0, method='RK45', t_eval=t_span,
            dense_output=True, events=collision, atol = 1e-13, rtol = 1e-13, 
            args=(params,)
        )
        
        t_temp1 = sol.t
        m, n = np.shape(sol.y)
        z_temp1 = np.zeros((n, m))
        z_temp1 = sol.y.T
        
        t_temp1 = tf + t_temp1
        tf = t_temp1[-1]
        
        ### collect reaction forces ###
        for i in range(len(t_temp1)):
            _, _, _, P_LA, P_RA, tau = single_stance_helper(t_temp1[i], z_temp1[i,:], params)
            if i == 0:
                P_LA_all[0] = P_LA
                P_RA_all[0] = P_RA
                Torque[0] = tau
            else:
                P_LA_all = np.vstack( (P_LA_all, P_LA) )
                P_RA_all = np.vstack( (P_RA_all, P_RA) )
                Torque = np.vstack( (Torque, tau) )
        
        ### foot strike: before to after foot strike ###
        params.P = params.Impulse
        
        z_plus, P_LA, P_RA = footstrike( t_temp1[-1], z_temp1[-1,:], params )
        
        ### swap legs ###
        if params.stance_foot == 'right':
            params.stance_foot = 'left'
        elif params.stance_foot == 'left':
            params.stance_foot = 'right'
            
        ### after foot strike to midstance ###
        z0 = z_plus
        
        t0, t1 = 0, 2
        t_span = np.linspace(t0, t1)
        
        x, xd, y, yd, z, zd, \
            phi, phid, theta, thetad, psi, psid, \
            phi_lh, phi_lhd, theta_lh, theta_lhd, \
            psi_lh, psi_lhd, theta_lk, theta_lkd, \
            phi_rh, phi_rhd, theta_rh, theta_rhd, \
            psi_rh, psi_rhd, theta_rk, theta_rkd = z0

        params.t0 = 0
        params.tf = 0.2
        
        if params.stance_foot == 'right':
            params.s0 = np.array([phi, theta, psi, phi_lh, theta_lh, psi_lh, theta_lk, theta_rk])
            params.v0 = np.array([phid, thetad, psid, phi_lhd, theta_lhd, psi_lhd, theta_lkd, theta_rkd])
            params.sf = np.array([0, 0, 0, 0, 0, 0, params.stepAngle, 0])
        elif params.stance_foot == 'left':
            params.s0 = np.array([phi, theta, psi, phi_rh, theta_rh, psi_rh, theta_lk, theta_rk])
            params.v0 = np.array([phid, thetad, psid, phi_rhd, theta_rhd, psi_rhd, theta_lkd, theta_rkd])
            params.sf = np.array([0, 0, 0, 0, 0, 0, params.stepAngle, 0])
        
        params.vf = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        params.a0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        params.af = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        midstance.terminal = True
        midstance.direction = 0
        
        sol = solve_ivp(
            single_stance, [t0, t1], z0, method='RK45', t_eval=t_span,
            dense_output=True, events=midstance, atol = 1e-13, rtol = 1e-13, 
            args=(params,)
        )
        
        t_temp2 = sol.t
        m, n = np.shape(sol.y)
        z_temp2 = np.zeros((n, m))
        z_temp2 = sol.y.T
        
        ### collect reaction forces ###
        for i in range(1, len(t_temp1)):
            _, _, _, P_LA, P_RA, tau = single_stance_helper(t_temp1[i], z_temp1[i,:], params)
            P_LA_all = np.vstack( (P_LA_all, P_LA) )
            P_RA_all = np.vstack( (P_RA_all, P_RA) )
            Torque = np.vstack( (Torque, tau) )
        
        t_temp2 = tf + t_temp1
        tf = t_temp2[-1]
        
        z0 = z_temp2[-1,:]
        
        t_ode = np.concatenate( (t, t_temp1, t_temp2), axis=0)
        z_ode = np.concatenate( (z, z_temp1, z_temp2), axis=0)
    
    if steps == 1:
        return z_ode[-1][6:]
    else:
        return z_ode, t_ode, P_LA_all, P_RA_all, Torque