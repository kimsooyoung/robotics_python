import numpy as np

from fourlinkchain_dynamics import fourlinkchain_dynamics

def controller(z, t, params, q1_refs, q2_refs):
    
    leg = params.leg
    
    q1, u1 = z[0], z[1]
    q2, u2 = z[2], z[3]
    q3, u3 = z[4], z[5]
    q4, u4 = z[6], z[7]
    
    Kp1, Kp2 = params.Kp1, params.Kp2
    Kd1, Kd2 = params.Kd1, params.Kd2
    
    A, b, J, Jdot = fourlinkchain_dynamics(z, params)
    
    if params.leg == 'minitaur' or params.leg == 'atrias':
        qdot = np.array([u1, u3, u2, u4])
        theta = np.array([[q1, q3]])
        thetadot = np.array([[u1, u3]])
    elif params.leg == 'digit':
        qdot = np.array([u1, u4, u2, u3])
        theta = np.array([[q1, q4]])
        thetadot = np.array([[u1, u4]])

    bigA = np.block([
        [A, -J.T],
        [J, np.zeros((2,2))]
    ])

    bigB = np.block([
        [ b ],
        [ np.reshape(-Jdot @ qdot.T, (2, 1)) ]
    ])

    q1_ref, q1d_ref, q1dd_ref = q1_refs
    q2_ref, q2d_ref, q2dd_ref = q2_refs
    
    theta_ref = np.array([
        [q1_ref],
        [q2_ref]
    ])
    
    thetadot_ref = np.array([
        [q1d_ref],
        [q2d_ref]
    ])

    thetaddot_ref = np.array([
        [q1dd_ref],
        [q2dd_ref]
    ])
    
    Kp = np.array([
        [Kp1, 0],
        [0, Kp2]
    ])
    
    Kd = np.array([
        [Kd1, 0],
        [0, Kd2]
    ])
    
    # A11 = np.zeros( (2,2) )
    # A12 = np.zeros( (2,4) )
    # A21 = np.zeros( (4,2) )
    # A22 = np.zeros( (4,4) )
    
    # B1 = np.zeros( (2,1) )
    # B2 = np.zeros( (4,1) )
    
    A11 = bigA[:2,:2]
    A12 = bigA[:2:,2:]
    A21 = bigA[2:,:2]
    A22 = bigA[2:,2:]
    
    # print(f"bigA : {bigA}")
    # print(f"A11 : {A11}")
    # print(f"A12 : {A12}")
    # print(f"A21 : {A21}")
    # print(f"A22 : {A22}")
    
    b1 = bigB[:2]
    b2 = bigB[2:]
    
    invA22 = np.linalg.inv(A22)
    Atil = A11 - A12 @ invA22 @ A21
    btil = b1 - A12 @ invA22 @ b2
    
    # print("thetaddot_ref")
    # print(thetaddot_ref)
    # print("theta.T - theta_ref")
    # print(theta.T - theta_ref)
    # print("theta.T")
    # print(theta.T)
    # print("theta_ref")
    # print(theta_ref)
    # print("Kp @ ( theta.T - theta_ref )")
    # print(Kp @ ( theta.T - theta_ref ))

    T = Atil @ ( thetaddot_ref - Kp @ ( theta.T - theta_ref ) - Kd @ ( thetadot.T - thetadot_ref ) ) - btil
    
    # print(T)
    # print(T[0][0])
    # print(T[1][0])
    
    return T[0][0], T[1][0]