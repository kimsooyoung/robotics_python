import numpy as np

def traj(t, t0, tf, s0, sf, v0, vf, a0, af):
    
    A = np.array([
        [1, t0, t0**2, t0**3, t0**4, t0**5],
        [1, tf, tf**2, tf**3, tf**4, tf**5],
        [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
        [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
        [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
        [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]
    ], dtype=np.float64)
    
    b = np.array([
        s0, sf, v0, vf, a0, af
    ], dtype=np.float64)
    
    Ainv = np.linalg.inv(A)
    X = Ainv.dot(b.T)
    # X = np.linalg.solve(A, b.T)
    
    # print("\nX = ")
    # print(X)
    # print("\n")
    
    c0 = X[0]; c1 = X[1]; c2 = X[2]; c3 = X[3]; c4 = X[4]; c5 = X[5];
    
    s = c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5
    v = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3 + 5*c5*t**4
    a = 2*c2 + 6*c3*t + 12*c4*t**2 + 20*c5*t**3
    
    # print(s, v, a)
    
    return s, v, a    
    