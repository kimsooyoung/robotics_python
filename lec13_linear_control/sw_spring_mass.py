import control
import numpy as np

class Parameters():

    def __init__(self):
        
        # spring-mass system
        self.m1 = 1
        self.m2 = 1 
        self.k1 = 2
        self.k2 = 2

        # animation params
        self.pause = .05
        self.fps = 30

if __name__=="__main__":

    params = Parameters()

    m1, m2 = params.m1, params.m2
    k1, k2 = params.k1, params.k2 

    # x = A*x + B*u
    A = np.array([
        [0,0,1,0],
        [0,0,0,1],
        [-(k1+k2)/m1,k2/m1,0,0],
        [k2/m2,-k2/m2,0,0]
    ])

    B = np.array([
        [0,0],
        [0,0],
        [-1/m1,0],
        [1/m1,1/m2],
    ])
    
    C = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,0],
        [0,0,0,0]
    ])

    # 1. calculate pole
    eigVal, eigVec = np.linalg.eig(A)
    print(f"eigVal: {eigVal}")

    # 2. controllability and observability 
    C_o = control.ctrb(A, B)
    print(f"C_o rank: {np.linalg.matrix_rank(C_o)}")
    
    Q_o = control.obsv(A, C)
    print(f"Q_o rank: {np.linalg.matrix_rank(Q_o)}")
    
    # 3. pole placement
    p = [-1,-2,-3,-4]
    k = control.place(A,B,p)
    print(f"k: {k}")
    eigVal_p, eigVec_p = np.linalg.eig(A - B@k)
    print(f"new eigVal, eigVec: \n {eigVal_p} \n {eigVec_p}")

    # 4. LQR
    # A: 4*4 matrix => Q: 4*4
    # u: 2*1 matrix => R: 2*2
    
    # Q >> R => aggressive controller
    Q = np.eye(4)
    # R = 0.1 * np.eye(2)
    # R = 0.0001 * np.eye(2)
    k,s,e = control.lqr(A,B,Q,R)
    print(f"K: {k}")
    print(f"S: {s}")
    print(f"E: {e}")
