import numpy as np 

def cos(angle): 
    return np.cos(angle) 

def sin(angle): 
    return np.sin(angle) 

def nlink_rhs(z, t, params): 

    m_0 = params.m1; I_0 = params.I1
    c_0 = params.c1; l_0 = params.l1;
    m_1 = params.m2; I_1 = params.I2
    c_1 = params.c2; l_1 = params.l2;
    m_2 = params.m3; I_2 = params.I3
    c_2 = params.c3; l_2 = params.l3;
    g = params.g

    q_0, u_0 = z[0], z[1]
    q_1, u_1 = z[2], z[3]
    q_2, u_2 = z[4], z[5] 

    M11 = 1.0*I_0 + 1.0*I_1 + 1.0*I_2 + c_0**2*m_0 + m_1*(c_1**2 + 2*c_1*l_0*cos(q_1) + l_0**2) + m_2*(c_2**2 + 2*c_2*l_0*cos(q_1 + q_2) + 2*c_2*l_1*cos(q_2) + l_0**2 + 2*l_0*l_1*cos(q_1) + l_1**2) 

    M12 = 1.0*I_1 + 1.0*I_2 + c_1*m_1*(c_1 + l_0*cos(q_1)) + m_2*(c_2**2 + c_2*l_0*cos(q_1 + q_2) + 2*c_2*l_1*cos(q_2) + l_0*l_1*cos(q_1) + l_1**2) 

    M13 = 1.0*I_2 + c_2*m_2*(c_2 + l_0*cos(q_1 + q_2) + l_1*cos(q_2)) 

    M21 = 1.0*I_1 + 1.0*I_2 + c_1*m_1*(c_1 + l_0*cos(q_1)) + m_2*(c_2**2 + c_2*l_0*cos(q_1 + q_2) + 2*c_2*l_1*cos(q_2) + l_0*l_1*cos(q_1) + l_1**2) 

    M22 = 1.0*I_1 + 1.0*I_2 + c_1**2*m_1 + m_2*(c_2**2 + 2*c_2*l_1*cos(q_2) + l_1**2) 

    M23 = 1.0*I_2 + c_2*m_2*(c_2 + l_1*cos(q_2)) 

    M31 = 1.0*I_2 + c_2*m_2*(c_2 + l_0*cos(q_1 + q_2) + l_1*cos(q_2)) 

    M32 = 1.0*I_2 + c_2*m_2*(c_2 + l_1*cos(q_2)) 

    M33 = 1.0*I_2 + c_2**2*m_2 


    C1 = -2.0*c_1*l_0*m_1*u_0*u_1*sin(q_1) - 1.0*c_1*l_0*m_1*u_1**2*sin(q_1) - 2.0*c_2*l_0*m_2*u_0*u_1*sin(q_1 + q_2) - 2.0*c_2*l_0*m_2*u_0*u_2*sin(q_1 + q_2) - 1.0*c_2*l_0*m_2*u_1**2*sin(q_1 + q_2) - 2.0*c_2*l_0*m_2*u_1*u_2*sin(q_1 + q_2) - 1.0*c_2*l_0*m_2*u_2**2*sin(q_1 + q_2) - 2.0*c_2*l_1*m_2*u_0*u_2*sin(q_2) - 2.0*c_2*l_1*m_2*u_1*u_2*sin(q_2) - 1.0*c_2*l_1*m_2*u_2**2*sin(q_2) - 2.0*l_0*l_1*m_2*u_0*u_1*sin(q_1) - 1.0*l_0*l_1*m_2*u_1**2*sin(q_1) 

    C2 = 1.0*c_1*l_0*m_1*u_0**2*sin(q_1) + 1.0*c_2*l_0*m_2*u_0**2*sin(q_1 + q_2) - 2.0*c_2*l_1*m_2*u_0*u_2*sin(q_2) - 2.0*c_2*l_1*m_2*u_1*u_2*sin(q_2) - 1.0*c_2*l_1*m_2*u_2**2*sin(q_2) + 1.0*l_0*l_1*m_2*u_0**2*sin(q_1) 

    C3 = c_2*m_2*(l_0*u_0**2*sin(q_1 + q_2) + l_1*u_0**2*sin(q_2) + 2*l_1*u_0*u_1*sin(q_2) + l_1*u_1**2*sin(q_2)) 


    G1 = g*(c_0*m_0*sin(q_0) + m_1*(c_1*sin(q_0 + q_1) + l_0*sin(q_0)) + m_2*(c_2*sin(q_0 + q_1 + q_2) + l_0*sin(q_0) + l_1*sin(q_0 + q_1))) 

    G2 = g*(c_1*m_1*sin(q_0 + q_1) + m_2*(c_2*sin(q_0 + q_1 + q_2) + l_1*sin(q_0 + q_1))) 

    G3 = c_2*g*m_2*sin(q_0 + q_1 + q_2) 


    A = np.array([ 
        [M11, M12, M13],
        [M21, M22, M23],
        [M31, M32, M33]
    ]) 

    b = -np.array([ 
        [C1 + G1],
        [C2 + G2],
        [C3 + G3]
    ]) 

    x = np.linalg.solve(A, b)

    output = np.array([
        u_0, x[0,0],
        u_1, x[1,0],
        u_2, x[2,0]
    ])

    return output 

