import numpy as np 

from controller import controller
from fourlinkchain_dynamics import fourlinkchain_dynamics

def fourlinkchain_rhs(z, t, params, q1_refs, q2_refs):

    q1, u1 = z[0], z[1] 
    q2, u2 = z[2], z[3] 
    q3, u3 = z[4], z[5] 
    q4, u4 = z[6], z[7] 

    A, b, J, Jdot = fourlinkchain_dynamics(z, params)

    T_first, T_second = controller(z, t, params, q1_refs, q2_refs)

    if params.leg == 'minitaur' or params.leg == 'atrias':
        qdot = np.array([u1, u3, u2, u4])
    elif params.leg == 'digit':
        qdot = np.array([u1, u4, u2, u3])

    T = np.array([[T_first, T_second, 0, 0]])

    bigA = np.block([
        [A, -J.T],
        [J, np.zeros((2,2))]
    ])

    bigB = np.block([
        [ b + T.T ],
        [ np.reshape(-Jdot @ qdot.T, (2, 1)) ]
    ])

    x = np.linalg.solve(bigA, bigB)

    if params.leg == 'minitaur' or params.leg == 'atrias':
        output = np.array([u1, x[0,0], u2, x[2,0], u3, x[1,0], u4, x[3,0]])
    elif params.leg == 'digit':
        output = np.array([u1, x[0,0], u2, x[2,0], u3, x[3,0], u4, x[1,0]])

    return output
