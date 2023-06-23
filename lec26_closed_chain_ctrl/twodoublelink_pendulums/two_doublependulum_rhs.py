import numpy as np 

def cos(angle): 
    return np.cos(angle) 

def sin(angle): 
    return np.sin(angle) 

def two_double_pendulum(z, t, params): 

    q1, u1 = z[0], z[1] 
    q2, u2 = z[2], z[3] 
    q3, u3 = z[4], z[5] 
    q4, u4 = z[6], z[7] 

    g, lx, ly = params.g, params.lx, params.ly 
    m1, m2, m3, m4 = params.m1, params.m2, params.m3, params.m4 
    I1, I2, I3, I4 = params.I1, params.I2, params.I3, params.I4 
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4 

    M11 = 1.0*I1 + 1.0*I2 + 0.25*l1**2*m1 + 0.5*m2*(2.0*l1**2 + 2.0*l1*l2*cos(q2) + 0.5*l2**2) 

    M12 = 1.0*I2 + 0.25*l2*m2*(2*l1*cos(q2) + l2) 

    M13 = 0 

    M14 = 0 

    M21 = 1.0*I2 + 0.5*l2*m2*(1.0*l1*cos(q2) + 0.5*l2) 

    M22 = 1.0*I2 + 0.25*l2**2*m2 

    M23 = 0 

    M24 = 0 

    M31 = 0 

    M32 = 0 

    M33 = 1.0*I3 + 1.0*I4 + 0.25*l3**2*m3 + 0.5*m4*(2.0*l3**2 + 2.0*l3*l4*cos(q4) + 0.5*l4**2) 

    M34 = 1.0*I4 + 0.25*l4*m4*(2*l3*cos(q4) + l4) 

    M41 = 0 

    M42 = 0 

    M43 = 1.0*I4 + 0.5*l4*m4*(1.0*l3*cos(q4) + 0.5*l4) 

    M44 = 1.0*I4 + 0.25*l4**2*m4 

    C1 = -l1*l2*m2*u2*(1.0*u1 + 0.5*u2)*sin(q2)

    C2 = 0.5*l1*l2*m2*u1**2*sin(q2)

    C3 = -l3*l4*m4*u4*(1.0*u3 + 0.5*u4)*sin(q4)

    C4 = 0.5*l3*l4*m4*u3**2*sin(q4)

    G1 = g*(0.5*l1*m1*sin(q1) + m2*(l1*sin(q1) + 0.5*l2*sin(q1 + q2)))

    G2 = 0.5*g*l2*m2*sin(q1 + q2)

    G3 = g*(0.5*l3*m3*sin(q3) + m4*(l3*sin(q3) + 0.5*l4*sin(q3 + q4)))

    G4 = 0.5*g*l4*m4*sin(q3 + q4)

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

    x = np.linalg.solve(A, b) 

    output = np.array([u1, x[0,0], u2, x[1,0], u3, x[2,0], u4, x[3,0]]) 

    return output 

