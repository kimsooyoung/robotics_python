import numpy as np
import matplotlib.pyplot as plt

# q1(t_init) = q_init / q1(t_mid) = q_mid
# q2(t_mid) = q_mid / q2(t_end) = q_end
# w1(t_init) = 0 / w2(t_end) = 0
# w1(t_mid) = w2(t_mid)
# a1(t_mid) = a2(t_mid)

def quinticpolytraj(q_init, q_mid, q_end, t_init, t_mid, t_end):

    t1 = t_init; t2 = t_init**2; t3 = t_init**3;
    tm1 = t_mid; tm2 = t_mid**2; tm3 = t_mid**3;
    te1 = t_end; te2 = t_end**2; te3 = t_end**3;

    A = np.matrix([
            [1, t1,   t2,  t3, 0, 0, 0, 0],
            [1, tm1, tm2, tm3, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, tm1, tm2, tm3],
            [0, 0, 0, 0, 1, te1, te2, te3],
            [0, 1, 2*t1, 3*t2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 2*te1, 3*te1],
            [0, 1, 2*tm1, 3*tm2, 0, -1, -2*tm1, -3*tm2],
            [0, 0, 2, 6*tm1, 0, 0, -2, -6*tm1]
        ])

    # b = np.matrix('0; 0.5; 0.5; 1; 0; 0.2; 0.2; 0')
    b = np.matrix([
        [q_init],
        [q_mid],
        [q_mid],
        [q_end],
        [0],
        [0],
        [0],
        [0]
    ])
    x = A.getI()*b

    # 이렇게 안된다. 
    # a10, a11, a12, a13 = x[:4,0]
    # a20, a21, a22, a23 = x[4:,0]

    a10 = x[0,0]; a11 = x[1,0]; a12 = x[2,0]; a13 = x[3,0];
    a20 = x[4,0]; a21 = x[5,0]; a22 = x[6,0]; a23 = x[7,0];

    tt1 = np.linspace(t_init, t_mid, 51)
    tt2 = np.linspace(t_mid, t_end, 51)

    q1_ref = a10 + a11*tt1 + a12*tt1**2 + a13*tt1**3
    q2_ref = a20 + a21*tt2 + a22*tt2**2 + a23*tt2**3

    q1d_ref = a11 + 2*a12*tt1 + 3*a13*tt1**2
    q2d_ref = a21 + 2*a22*tt2 + 3*a23*tt2**2

    q1dd_ref = 2*a12 + 6*a13*tt1
    q2dd_ref = 2*a22 + 6*a23*tt2
    
    q_ref = np.concatenate( (q1_ref, q2_ref[1:]), axis=0 )
    qd_ref = np.concatenate( (q1d_ref, q2d_ref[1:]), axis=0 )
    qdd_ref = np.concatenate( (q1dd_ref, q2dd_ref[1:]), axis=0 )
    
    t_ref = np.concatenate( (tt1, tt2[1:]), axis=0 )
    
    return q_ref, qd_ref, qdd_ref, t_ref
