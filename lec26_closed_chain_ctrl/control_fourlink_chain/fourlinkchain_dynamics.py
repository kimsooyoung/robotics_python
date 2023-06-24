import numpy as np 

def cos(angle): 
    return np.cos(angle) 

def sin(angle): 
    return np.sin(angle) 

def fourlinkchain_dynamics(z, params): 

    q1, u1 = z[0], z[1] 
    q2, u2 = z[2], z[3] 
    q3, u3 = z[4], z[5] 
    q4, u4 = z[6], z[7] 

    g, lx, ly = params.g, params.lx, params.ly 
    m1, m2, m3, m4 = params.m1, params.m2, params.m3, params.m4 
    I1, I2, I3, I4 = params.I1, params.I2, params.I3, params.I4 
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4 

    M11 = 1.0*I1 + 1.0*I2 + 0.25*l1**2*m1 + 0.5*m2*(2.0*l1**2 + 2.0*l1*l2*cos(q2) + 0.5*l2**2) 

    M12 = 0 

    M13 = 1.0*I2 + 0.25*l2*m2*(2*l1*cos(q2) + l2) 

    M14 = 0 

    M21 = 0 

    M22 = 1.0*I4 + 0.25*l4**2*m4 

    M23 = 0 

    M24 = 1.0*I4 + 0.5*l4*m4*(1.0*l3*cos(q4) + 0.5*l4) 

    M31 = 1.0*I2 + 0.5*l2*m2*(1.0*l1*cos(q2) + 0.5*l2) 

    M32 = 0 

    M33 = 1.0*I2 + 0.25*l2**2*m2 

    M34 = 0 

    M41 = 0 

    M42 = 1.0*I4 + 0.25*l4*m4*(2*l3*cos(q4) + l4) 

    M43 = 0 

    M44 = 1.0*I3 + 1.0*I4 + 0.25*l3**2*m3 + 0.5*m4*(2.0*l3**2 + 2.0*l3*l4*cos(q4) + 0.5*l4**2) 

    C1 = -1.0*l1*l2*m2*u1*u2*sin(q2)

    C2 = 0

    C3 = 0.5*l1*l2*m2*u1**2*sin(q2)

    C4 = 0

    G1 = 0.5*g*l1*m1*sin(q1) + 1.0*g*l1*m2*sin(q1) + 0.5*g*l2*m2*sin(q1 + q2) - 0.5*l1*l2*m2*u2**2*sin(q2)

    G2 = 0.5*l4*m4*(g*sin(q3 + q4) + l3*u3**2*sin(q4))

    G3 = 0.5*g*l2*m2*sin(q1 + q2)

    G4 = 0.5*g*l3*m3*sin(q3) + 1.0*g*l3*m4*sin(q3) + 0.5*g*l4*m4*sin(q3 + q4) - 1.0*l3*l4*m4*u3*u4*sin(q4) - 0.5*l3*l4*m4*u4**2*sin(q4)

    J11 = l1*cos(q1) + l2*cos(q1 + q2)

    J12 = -l4*cos(q3 + q4)

    J13 = l2*cos(q1 + q2)

    J14 = -l3*cos(q3) - l4*cos(q3 + q4)

    J21 = l1*sin(q1) + l2*sin(q1 + q2)

    J22 = -l4*sin(q3 + q4)

    J23 = l2*sin(q1 + q2)

    J24 = -l3*sin(q3) - l4*sin(q3 + q4)

    Jdot11 = -l2*u2*sin(q1 + q2) + u1*(-l1*sin(q1) - l2*sin(q1 + q2))

    Jdot12 = l4*u3*sin(q3 + q4) + l4*u4*sin(q3 + q4)

    Jdot13 = -l2*u1*sin(q1 + q2) - l2*u2*sin(q1 + q2)

    Jdot14 = l4*u4*sin(q3 + q4) + u3*(l3*sin(q3) + l4*sin(q3 + q4))

    Jdot21 = l2*u2*cos(q1 + q2) + u1*(l1*cos(q1) + l2*cos(q1 + q2))

    Jdot22 = -l4*u3*cos(q3 + q4) - l4*u4*cos(q3 + q4)

    Jdot23 = l2*u1*cos(q1 + q2) + l2*u2*cos(q1 + q2)

    Jdot24 = -l4*u4*cos(q3 + q4) + u3*(-l3*cos(q3) - l4*cos(q3 + q4))

    A = np.array([ 
        [M11, M12, M13, M14], 
        [M21, M22, M23, M24], 
        [M31, M32, M33, M34], 
        [M41, M42, M43, M44] 
    ]) 

    b = -np.array([ 
        [C1 + G1], 
        [C2 + G2], 
        [C3 + G3], 
        [C4 + G4] 
    ]) 

    J = np.array([
        [J11, J12, J13, J14],
        [J21, J22, J23, J24],
    ])

    Jdot = np.array([
        [Jdot11, Jdot12, Jdot13, Jdot14],
        [Jdot21, Jdot22, Jdot23, Jdot24],
    ])

    return A, b, J, Jdot
