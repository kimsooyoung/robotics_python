import numpy as np

def cos(theta):
    return np.cos(theta)

def sin(theta):
    return np.sin(theta)

def foot_impulse(P,l2,phi,phi_lh,phi_rh,psi_lh,psi_rh,psi,theta,theta_lh,theta_lk,theta_rh,theta_rk):
    
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
    t15 = sin(phi_rh)
    t16 = sin(psi_lh)
    t17 = sin(psi_rh)
    t18 = sin(psi)
    t19 = sin(theta)
    t20 = sin(theta_lh)
    t21 = sin(theta_lk)
    t22 = sin(theta_rh)
    t23 = sin(theta_rk)
    t24 = 1.0 / l2
    
    mt1 = P*t24*(l2*t3*t9*t10*t19-l2*t3*t19*t20*t21-l2*t5*t7*t8*t9*t21-l2*t5*t7*t8*t10*t20+l2*t8*t9*t16*t18*t21+l2*t8*t10*t16*t18*t20+l2*t5*t8*t9*t10*t14*t18+l2*t7*t8*t9*t10*t14*t16-l2*t5*t8*t14*t18*t20*t21-l2*t7*t8*t14*t16*t20*t21)
    mt2 = -P*t24*(l2*t3*t8*t9*t10*t13+l2*t2*t5*t9*t18*t21+l2*t2*t5*t10*t18*t20+l2*t2*t7*t9*t16*t21+l2*t2*t7*t10*t16*t20-l2*t3*t8*t13*t20*t21+l2*t2*t5*t7*t9*t10*t14-l2*t2*t5*t7*t14*t20*t21-l2*t2*t9*t10*t14*t16*t18+l2*t5*t7*t9*t13*t19*t21+l2*t5*t7*t10*t13*t19*t20+l2*t2*t14*t16*t18*t20*t21-l2*t9*t13*t16*t18*t19*t21-l2*t10*t13*t16*t18*t19*t20-l2*t5*t9*t10*t13*t14*t18*t19-l2*t7*t9*t10*t13*t14*t16*t19+l2*t5*t13*t14*t18*t19*t20*t21+l2*t7*t13*t14*t16*t19*t20*t21)
    mt3 = -P*t24*(-l2*t2*t3*t8*t9*t10+l2*t2*t3*t8*t20*t21+l2*t5*t9*t13*t18*t21+l2*t5*t10*t13*t18*t20+l2*t7*t9*t13*t16*t21+l2*t7*t10*t13*t16*t20+l2*t5*t7*t9*t10*t13*t14-l2*t2*t5*t7*t9*t19*t21-l2*t2*t5*t7*t10*t19*t20-l2*t5*t7*t13*t14*t20*t21-l2*t9*t10*t13*t14*t16*t18+l2*t2*t9*t16*t18*t19*t21+l2*t2*t10*t16*t18*t19*t20+l2*t13*t14*t16*t18*t20*t21+l2*t2*t5*t9*t10*t14*t18*t19+l2*t2*t7*t9*t10*t14*t16*t19-l2*t2*t5*t14*t18*t19*t20*t21-l2*t2*t7*t14*t16*t19*t20*t21)
    I_LA = np.array([mt1,mt2,mt3])
    
    mt4 = -P*t24*(-l2*t4*t11*t12*t19+l2*t4*t19*t22*t23+l2*t6*t7*t8*t11*t23+l2*t6*t7*t8*t12*t22+l2*t8*t11*t17*t18*t23+l2*t8*t12*t17*t18*t22+l2*t6*t8*t11*t12*t15*t18-l2*t7*t8*t11*t12*t15*t17-l2*t6*t8*t15*t18*t22*t23+l2*t7*t8*t15*t17*t22*t23)
    mt5 = -P*t24*(l2*t4*t8*t11*t12*t13+l2*t2*t6*t11*t18*t23+l2*t2*t6*t12*t18*t22-l2*t2*t7*t11*t17*t23-l2*t2*t7*t12*t17*t22-l2*t4*t8*t13*t22*t23-l2*t2*t6*t7*t11*t12*t15+l2*t2*t6*t7*t15*t22*t23-l2*t2*t11*t12*t15*t17*t18+l2*t6*t7*t11*t13*t19*t23+l2*t6*t7*t12*t13*t19*t22+l2*t2*t15*t17*t18*t22*t23+l2*t11*t13*t17*t18*t19*t23+l2*t12*t13*t17*t18*t19*t22+l2*t6*t11*t12*t13*t15*t18*t19-l2*t7*t11*t12*t13*t15*t17*t19-l2*t6*t13*t15*t18*t19*t22*t23+l2*t7*t13*t15*t17*t19*t22*t23)
    mt6 = P*t24*(l2*t2*t4*t8*t11*t12-l2*t2*t4*t8*t22*t23-l2*t6*t11*t13*t18*t23-l2*t6*t12*t13*t18*t22+l2*t7*t11*t13*t17*t23+l2*t7*t12*t13*t17*t22+l2*t6*t7*t11*t12*t13*t15+l2*t2*t6*t7*t11*t19*t23+l2*t2*t6*t7*t12*t19*t22-l2*t6*t7*t13*t15*t22*t23+l2*t11*t12*t13*t15*t17*t18+l2*t2*t11*t17*t18*t19*t23+l2*t2*t12*t17*t18*t19*t22-l2*t13*t15*t17*t18*t22*t23+l2*t2*t6*t11*t12*t15*t18*t19-l2*t2*t7*t11*t12*t15*t17*t19-l2*t2*t6*t15*t18*t19*t22*t23+l2*t2*t7*t15*t17*t19*t22*t23)
    I_RA = np.array([mt4,mt5,mt6])
    
    return I_LA,I_RA
