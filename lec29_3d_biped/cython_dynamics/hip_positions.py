import numpy as np

def cos(theta):
    return np.cos(theta)

def sin(theta):
    return np.sin(theta)

def hip_positions(l1, l2, phi, phi_lh, phi_rh, psi_lh, psi_rh, psi, theta, theta_lh, theta_lk, theta_rh, theta_rk, w):
    
    # matlab result
    t2 = cos(phi)
    t3 = cos(phi_lh)
    t4 = cos(phi_rh)
    t5 = cos(psi_lh)
    t6 = cos(psi_rh)
    t7 = cos(psi)
    t8 = cos(theta)
    t9 = cos(theta_lh)
    t10 = cos(theta_lk)
    t11 = cos(theta_rh)
    t12 = cos(theta_rk)
    t13 = sin(phi)
    t14 = sin(phi_lh)
    t15 = sin(phi_rh)
    t16 = sin(psi_lh)
    t17 = sin(psi_rh)
    t18 = sin(psi)
    t19 = sin(theta)
    t20 = sin(theta_lh)
    t21 = sin(theta_lk)
    t22 = sin(theta_rh)
    t23 = sin(theta_rk)
    t24 = t2*t7*w
    t25 = t7*t13*w
    t26 = t8*t18*w
    t27 = t2*t18*t19*w
    t28 = t13*t18*t19*w
    
    et1 = -t24+t28-l1*t3*t8*t9*t13-l1*t2*t5*t18*t20-l1*t2*t7*t16*t20-l1*t2*t5*t7*t9*t14-l2*t3*t8*t9*t10*t13-l2*t2*t5*t9*t18*t21-l2*t2*t5*t10*t18*t20-l2*t2*t7*t9*t16*t21-l2*t2*t7*t10*t16*t20+l1*t2*t9*t14*t16*t18-l1*t5*t7*t13*t19*t20+l2*t3*t8*t13*t20*t21+l1*t13*t16*t18*t19*t20-l2*t2*t5*t7*t9*t10*t14+l2*t2*t5*t7*t14*t20*t21+l2*t2*t9*t10*t14*t16*t18-l2*t5*t7*t9*t13*t19*t21-l2*t5*t7*t10*t13*t19*t20+l1*t5*t9*t13*t14*t18*t19+l1*t7*t9*t13*t14*t16*t19-l2*t2*t14*t16*t18*t20*t21+l2*t9*t13*t16*t18*t19*t21;
    et2 = l2*t10*t13*t16*t18*t19*t20+l2*t5*t9*t10*t13*t14*t18*t19+l2*t7*t9*t10*t13*t14*t16*t19-l2*t5*t13*t14*t18*t19*t20*t21-l2*t7*t13*t14*t16*t19*t20*t21;
    et3 = -t25-t27+l1*t2*t3*t8*t9-l1*t5*t13*t18*t20-l1*t7*t13*t16*t20+l2*t2*t3*t8*t9*t10-l1*t5*t7*t9*t13*t14+l1*t2*t5*t7*t19*t20-l2*t2*t3*t8*t20*t21-l2*t5*t9*t13*t18*t21-l2*t5*t10*t13*t18*t20-l2*t7*t9*t13*t16*t21-l2*t7*t10*t13*t16*t20+l1*t9*t13*t14*t16*t18-l1*t2*t16*t18*t19*t20-l2*t5*t7*t9*t10*t13*t14+l2*t2*t5*t7*t9*t19*t21+l2*t2*t5*t7*t10*t19*t20-l1*t2*t5*t9*t14*t18*t19-l1*t2*t7*t9*t14*t16*t19+l2*t5*t7*t13*t14*t20*t21+l2*t9*t10*t13*t14*t16*t18-l2*t2*t9*t16*t18*t19*t21-l2*t2*t10*t16*t18*t19*t20;
    et4 = -l2*t13*t14*t16*t18*t20*t21-l2*t2*t5*t9*t10*t14*t18*t19-l2*t2*t7*t9*t10*t14*t16*t19+l2*t2*t5*t14*t18*t19*t20*t21+l2*t2*t7*t14*t16*t19*t20*t21;
    
    pos_hip_l_stance = np.array([
        t26+l1*t3*t9*t19-l1*t5*t7*t8*t20+l2*t3*t9*t10*t19+l1*t8*t16*t18*t20-l2*t3*t19*t20*t21-l2*t5*t7*t8*t9*t21-l2*t5*t7*t8*t10*t20+l1*t5*t8*t9*t14*t18+l1*t7*t8*t9*t14*t16+l2*t8*t9*t16*t18*t21+l2*t8*t10*t16*t18*t20+l2*t5*t8*t9*t10*t14*t18+l2*t7*t8*t9*t10*t14*t16-l2*t5*t8*t14*t18*t20*t21-l2*t7*t8*t14*t16*t20*t21,
        et1+et2,
        et3+et4,
        -1.0
    ])
    
    et5 = t24-t28-l1*t4*t8*t11*t13-l1*t2*t6*t18*t22+l1*t2*t7*t17*t22+l1*t2*t6*t7*t11*t15-l2*t4*t8*t11*t12*t13-l2*t2*t6*t11*t18*t23-l2*t2*t6*t12*t18*t22+l2*t2*t7*t11*t17*t23+l2*t2*t7*t12*t17*t22+l1*t2*t11*t15*t17*t18-l1*t6*t7*t13*t19*t22+l2*t4*t8*t13*t22*t23-l1*t13*t17*t18*t19*t22+l2*t2*t6*t7*t11*t12*t15-l2*t2*t6*t7*t15*t22*t23+l2*t2*t11*t12*t15*t17*t18-l2*t6*t7*t11*t13*t19*t23-l2*t6*t7*t12*t13*t19*t22-l1*t6*t11*t13*t15*t18*t19+l1*t7*t11*t13*t15*t17*t19-l2*t2*t15*t17*t18*t22*t23-l2*t11*t13*t17*t18*t19*t23;
    et6 = -l2*t12*t13*t17*t18*t19*t22-l2*t6*t11*t12*t13*t15*t18*t19+l2*t7*t11*t12*t13*t15*t17*t19+l2*t6*t13*t15*t18*t19*t22*t23-l2*t7*t13*t15*t17*t19*t22*t23;
    et7 = t25+t27+l1*t2*t4*t8*t11-l1*t6*t13*t18*t22+l1*t7*t13*t17*t22+l2*t2*t4*t8*t11*t12+l1*t6*t7*t11*t13*t15+l1*t2*t6*t7*t19*t22-l2*t2*t4*t8*t22*t23-l2*t6*t11*t13*t18*t23-l2*t6*t12*t13*t18*t22+l2*t7*t11*t13*t17*t23+l2*t7*t12*t13*t17*t22+l1*t11*t13*t15*t17*t18+l1*t2*t17*t18*t19*t22+l2*t6*t7*t11*t12*t13*t15+l2*t2*t6*t7*t11*t19*t23+l2*t2*t6*t7*t12*t19*t22+l1*t2*t6*t11*t15*t18*t19-l1*t2*t7*t11*t15*t17*t19-l2*t6*t7*t13*t15*t22*t23+l2*t11*t12*t13*t15*t17*t18+l2*t2*t11*t17*t18*t19*t23+l2*t2*t12*t17*t18*t19*t22-l2*t13*t15*t17*t18*t22*t23+l2*t2*t6*t11*t12*t15*t18*t19;
    et8 = -l2*t2*t7*t11*t12*t15*t17*t19-l2*t2*t6*t15*t18*t19*t22*t23+l2*t2*t7*t15*t17*t19*t22*t23;
    pos_hip_r_stance = np.array([
        -t26+l1*t4*t11*t19-l1*t6*t7*t8*t22+l2*t4*t11*t12*t19-l1*t8*t17*t18*t22-l2*t4*t19*t22*t23-l2*t6*t7*t8*t11*t23-l2*t6*t7*t8*t12*t22-l1*t6*t8*t11*t15*t18+l1*t7*t8*t11*t15*t17-l2*t8*t11*t17*t18*t23-l2*t8*t12*t17*t18*t22-l2*t6*t8*t11*t12*t15*t18+l2*t7*t8*t11*t12*t15*t17+l2*t6*t8*t15*t18*t22*t23-l2*t7*t8*t15*t17*t22*t23,
        et5+et6,
        et7+et8,
        -1.0
    ])
    
    ### python result ###
    # pos_hip_l_stance_1 = l1*(1 - cos(theta_lk))*((-(-sin(psi)*cos(psi_lh)*cos(theta) - sin(psi_lh)*cos(psi)*cos(theta))*sin(phi_lh) + sin(theta)*cos(phi_lh))*cos(theta_lh) - (-sin(psi)*sin(psi_lh)*cos(theta) + cos(psi)*cos(psi_lh)*cos(theta))*sin(theta_lh)) + l1*((-(-sin(psi)*cos(psi_lh)*cos(theta) - sin(psi_lh)*cos(psi)*cos(theta))*sin(phi_lh) + sin(theta)*cos(phi_lh))*sin(theta_lh) + (-sin(psi)*sin(psi_lh)*cos(theta) + cos(psi)*cos(psi_lh)*cos(theta))*cos(theta_lh))*sin(theta_lk) - w*(1 - cos(phi_lh))*(-sin(psi)*cos(psi_lh)*cos(theta) - sin(psi_lh)*cos(psi)*cos(theta)) + w*(1 - cos(psi_lh))*sin(psi)*cos(theta) - w*((-sin(psi)*cos(psi_lh)*cos(theta) - sin(psi_lh)*cos(psi)*cos(theta))*cos(phi_lh) + sin(phi_lh)*sin(theta)) + w*sin(phi_lh)*sin(theta) - w*sin(psi_lh)*cos(psi)*cos(theta) - (-l1 - l2)*(-((-(-sin(psi)*cos(psi_lh)*cos(theta) - sin(psi_lh)*cos(psi)*cos(theta))*sin(phi_lh) + sin(theta)*cos(phi_lh))*sin(theta_lh) + (-sin(psi)*sin(psi_lh)*cos(theta) + cos(psi)*cos(psi_lh)*cos(theta))*cos(theta_lh))*sin(theta_lk) + ((-(-sin(psi)*cos(psi_lh)*cos(theta) - sin(psi_lh)*cos(psi)*cos(theta))*sin(phi_lh) + sin(theta)*cos(phi_lh))*cos(theta_lh) - (-sin(psi)*sin(psi_lh)*cos(theta) + cos(psi)*cos(psi_lh)*cos(theta))*sin(theta_lh))*cos(theta_lk))
    # pos_hip_l_stance_2 = l1*(1 - cos(theta_lk))*((-((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*cos(psi_lh) - (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_lh))*sin(phi_lh) - sin(phi)*cos(phi_lh)*cos(theta))*cos(theta_lh) - ((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*sin(psi_lh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*cos(psi_lh))*sin(theta_lh)) + l1*((-((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*cos(psi_lh) - (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_lh))*sin(phi_lh) - sin(phi)*cos(phi_lh)*cos(theta))*sin(theta_lh) + ((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*sin(psi_lh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*cos(psi_lh))*cos(theta_lh))*sin(theta_lk) - w*(1 - cos(phi_lh))*((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*cos(psi_lh) - (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_lh)) - w*(1 - cos(psi_lh))*(-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi)) - w*(((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*cos(psi_lh) - (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_lh))*cos(phi_lh) - sin(phi)*sin(phi_lh)*cos(theta)) - w*(sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_lh) - w*sin(phi)*sin(phi_lh)*cos(theta) - (-l1 - l2)*(-((-((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*cos(psi_lh) - (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_lh))*sin(phi_lh) - sin(phi)*cos(phi_lh)*cos(theta))*sin(theta_lh) + ((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*sin(psi_lh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*cos(psi_lh))*cos(theta_lh))*sin(theta_lk) + ((-((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*cos(psi_lh) - (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_lh))*sin(phi_lh) - sin(phi)*cos(phi_lh)*cos(theta))*cos(theta_lh) - ((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*sin(psi_lh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*cos(psi_lh))*sin(theta_lh))*cos(theta_lk))
    # pos_hip_l_stance_3 = l1*(1 - cos(theta_lk))*((-(-(sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_lh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*cos(psi_lh))*sin(phi_lh) + cos(phi)*cos(phi_lh)*cos(theta))*cos(theta_lh) - ((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*cos(psi_lh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*sin(psi_lh))*sin(theta_lh)) + l1*((-(-(sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_lh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*cos(psi_lh))*sin(phi_lh) + cos(phi)*cos(phi_lh)*cos(theta))*sin(theta_lh) + ((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*cos(psi_lh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*sin(psi_lh))*cos(theta_lh))*sin(theta_lk) - w*(1 - cos(phi_lh))*(-(sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_lh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*cos(psi_lh)) - w*(1 - cos(psi_lh))*(sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi)) - w*((-(sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_lh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*cos(psi_lh))*cos(phi_lh) + sin(phi_lh)*cos(phi)*cos(theta)) - w*(sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_lh) + w*sin(phi_lh)*cos(phi)*cos(theta) - (-l1 - l2)*(-((-(-(sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_lh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*cos(psi_lh))*sin(phi_lh) + cos(phi)*cos(phi_lh)*cos(theta))*sin(theta_lh) + ((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*cos(psi_lh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*sin(psi_lh))*cos(theta_lh))*sin(theta_lk) + ((-(-(sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_lh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*cos(psi_lh))*sin(phi_lh) + cos(phi)*cos(phi_lh)*cos(theta))*cos(theta_lh) - ((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*cos(psi_lh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*sin(psi_lh))*sin(theta_lh))*cos(theta_lk))
    # pos_hip_l_stance_4 = -1
    
    # pos_hip_r_stance_1 = l1*(1 - cos(theta_rk))*(((-sin(psi)*cos(psi_rh)*cos(theta) + sin(psi_rh)*cos(psi)*cos(theta))*sin(phi_rh) + sin(theta)*cos(phi_rh))*cos(theta_rh) - (sin(psi)*sin(psi_rh)*cos(theta) + cos(psi)*cos(psi_rh)*cos(theta))*sin(theta_rh)) + l1*(((-sin(psi)*cos(psi_rh)*cos(theta) + sin(psi_rh)*cos(psi)*cos(theta))*sin(phi_rh) + sin(theta)*cos(phi_rh))*sin(theta_rh) + (sin(psi)*sin(psi_rh)*cos(theta) + cos(psi)*cos(psi_rh)*cos(theta))*cos(theta_rh))*sin(theta_rk) + w*(1 - cos(phi_rh))*(-sin(psi)*cos(psi_rh)*cos(theta) + sin(psi_rh)*cos(psi)*cos(theta)) - w*(1 - cos(psi_rh))*sin(psi)*cos(theta) + w*((-sin(psi)*cos(psi_rh)*cos(theta) + sin(psi_rh)*cos(psi)*cos(theta))*cos(phi_rh) - sin(phi_rh)*sin(theta)) + w*sin(phi_rh)*sin(theta) - w*sin(psi_rh)*cos(psi)*cos(theta) - (-l1 - l2)*(-(((-sin(psi)*cos(psi_rh)*cos(theta) + sin(psi_rh)*cos(psi)*cos(theta))*sin(phi_rh) + sin(theta)*cos(phi_rh))*sin(theta_rh) + (sin(psi)*sin(psi_rh)*cos(theta) + cos(psi)*cos(psi_rh)*cos(theta))*cos(theta_rh))*sin(theta_rk) + (((-sin(psi)*cos(psi_rh)*cos(theta) + sin(psi_rh)*cos(psi)*cos(theta))*sin(phi_rh) + sin(theta)*cos(phi_rh))*cos(theta_rh) - (sin(psi)*sin(psi_rh)*cos(theta) + cos(psi)*cos(psi_rh)*cos(theta))*sin(theta_rh))*cos(theta_rk))
    # pos_hip_r_stance_2 = l1*(1 - cos(theta_rk))*((((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*cos(psi_rh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_rh))*sin(phi_rh) - sin(phi)*cos(phi_rh)*cos(theta))*cos(theta_rh) - (-(-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*sin(psi_rh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*cos(psi_rh))*sin(theta_rh)) + l1*((((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*cos(psi_rh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_rh))*sin(phi_rh) - sin(phi)*cos(phi_rh)*cos(theta))*sin(theta_rh) + (-(-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*sin(psi_rh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*cos(psi_rh))*cos(theta_rh))*sin(theta_rk) + w*(1 - cos(phi_rh))*((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*cos(psi_rh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_rh)) + w*(1 - cos(psi_rh))*(-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi)) + w*(((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*cos(psi_rh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_rh))*cos(phi_rh) + sin(phi)*sin(phi_rh)*cos(theta)) - w*(sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_rh) - w*sin(phi)*sin(phi_rh)*cos(theta) - (-l1 - l2)*(-((((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*cos(psi_rh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_rh))*sin(phi_rh) - sin(phi)*cos(phi_rh)*cos(theta))*sin(theta_rh) + (-(-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*sin(psi_rh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*cos(psi_rh))*cos(theta_rh))*sin(theta_rk) + ((((-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*cos(psi_rh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*sin(psi_rh))*sin(phi_rh) - sin(phi)*cos(phi_rh)*cos(theta))*cos(theta_rh) - (-(-sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*sin(psi_rh) + (sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))*cos(psi_rh))*sin(theta_rh))*cos(theta_rk))
    # pos_hip_r_stance_3 = l1*(1 - cos(theta_rk))*((((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_rh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*cos(psi_rh))*sin(phi_rh) + cos(phi)*cos(phi_rh)*cos(theta))*cos(theta_rh) - ((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*cos(psi_rh) - (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*sin(psi_rh))*sin(theta_rh)) + l1*((((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_rh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*cos(psi_rh))*sin(phi_rh) + cos(phi)*cos(phi_rh)*cos(theta))*sin(theta_rh) + ((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*cos(psi_rh) - (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*sin(psi_rh))*cos(theta_rh))*sin(theta_rk) + w*(1 - cos(phi_rh))*((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_rh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*cos(psi_rh)) + w*(1 - cos(psi_rh))*(sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi)) + w*(((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_rh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*cos(psi_rh))*cos(phi_rh) - sin(phi_rh)*cos(phi)*cos(theta)) - w*(sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_rh) + w*sin(phi_rh)*cos(phi)*cos(theta) - (-l1 - l2)*(-((((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_rh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*cos(psi_rh))*sin(phi_rh) + cos(phi)*cos(phi_rh)*cos(theta))*sin(theta_rh) + ((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*cos(psi_rh) - (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*sin(psi_rh))*cos(theta_rh))*sin(theta_rk) + ((((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*sin(psi_rh) + (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*cos(psi_rh))*sin(phi_rh) + cos(phi)*cos(phi_rh)*cos(theta))*cos(theta_rh) - ((sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi))*cos(psi_rh) - (sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*sin(psi_rh))*sin(theta_rh))*cos(theta_rk))
    # pos_hip_r_stance_4 = -1
    
    # pos_hip_l_stance = np.array([pos_hip_l_stance_1, pos_hip_l_stance_2, pos_hip_l_stance_3, pos_hip_l_stance_4])
    # pos_hip_r_stance = np.array([pos_hip_r_stance_1, pos_hip_r_stance_2, pos_hip_r_stance_3, pos_hip_r_stance_4])
    
    return pos_hip_l_stance, pos_hip_r_stance
