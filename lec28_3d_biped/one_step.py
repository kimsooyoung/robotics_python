from scipy.integrate import solve_ivp
from single_stance import single_stance

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
        
    z0 = np.hstack( [x, xd, y, yd, z, zd], z0 )
    
    t_ode = 0
    z_ode = z0
    tf = 0
    
    P_LA_all = []
    P_RA_all = []
    Torque = []
    
    for i in range(steps):
        
        t0, t1 = 0, 2
        x, xd, y, yd, z, zd, \
            phi, phid, theta, thetad, psi, psid, \
            phi_lh, phi_lhd, theta_lh, theta_lhd, \
            psi_lh, psi_lhd, theta_lk, theta_lkd, \
            phi_rh, phi_rhd, theta_rh, theta_rhd, \
            psi_rh, psi_rhd, theta_rk, theta_rkd = z0
            
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
            single_stance, [params.t0, params.tf], z0, method='RK45', t_eval=t,
            dense_output=True, events=collision, atol = 1e-13, rtol = 1e-13, 
            args=(params,)
        )
        
        
        ### collect reaction forces ###
        if i == 0: # only for the first step take the first index
        #     j = 1;
        #     [~,~,~,P_LA,P_RA,tau] = single_stanceMEX(t_temp1(j),z_temp1(j,:),parms);
        #     P_LA_all = [P_LA_all; P_LA];
        #     P_RA_all = [P_RA_all; P_RA];
        #     Torque = [Torque; tau'];
        # end
        # for j=2:length(t_temp1)
        #     [~,~,~,P_LA,P_RA,tau] = single_stanceMEX(t_temp1(j),z_temp1(j,:),parms);
        #     P_LA_all = [P_LA_all; P_LA];
        #     P_RA_all = [P_RA_all; P_RA];
        #     Torque = [Torque; tau'];
        #     %P_LA_all(j,:,i) = P_LA; %jth item at step i
        #     %P_RA_all(j,:,i) = P_RA;
        #     %Torque(j,:,i) = tau;
        # end
        # t_temp1 = tf+t_temp1;
        # tf = t_temp1(end);
        
        ### foot strike: before to after foot strike ###
        