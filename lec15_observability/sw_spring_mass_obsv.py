from matplotlib import pyplot as plt
import numpy as np
import control

class Paraemters():
    
    def __init__(self):
        self.m1, self.m2 = 1, 1
        self.k1, self.k2 = 2, 3
        
def dynamics(m1, m2, k1, k2):
    
    A = np.array([
        [0,0,1,0],
        [0,0,0,1],
        [-(k1/m1+k2/m1), k2/m1, 0, 0],
        [k2/m2, -k2/m2, 0, 0]
    ])
    
    B = np.array([
        [0,0],
        [0,0],
        [-1/m1, 0],
        [1/m2,1/m2]
    ])
    
    # observe velocity
    C = np.array([
        [0,0,1,0],
        [0,0,0,1]
    ])
    
    return A, B, C

if __name__=="__main__":
    
    params = Paraemters()
    m1, m2, k1, k2 = params.m1, params.m2, params.k1, params.k2
    
    A, B, C = dynamics(m1, m2, k1, k2)
    
    # 1. compute eigenvalues of uncontrolled system
    eigVal, eigVec = np.linalg.eig(A)
    print(f'eig-vals (uncontrolled)')
    print(eigVal, '\n')
    
    # 2. compute observability of the system (2 ways)
    # 2.1. compute observability matrix
    Ob = control.obsv(A,C)
    print("control.obsv(A,C)")
    print(Ob)
    # print(f'rank={np.linalg.matrix_rank(Ob)}')
    
    # 2.2. compute observability matrix using transpose of controllability matrix
    Ob_trans = control.ctrb(A.T, C.T)
    print("control.ctrb(A.T, C.T)")
    print(Ob_trans.T)
    
    # 3. observability stability
    rank = np.linalg.matrix_rank(Ob)
    print("Rank of Ob")
    print(rank)
    
    # 4. pole replacement for stable observability
    p = np.array([-0.5, -0.6, -0.65, -6])
    L_trans = control.place(A.T, C.T, p)
    L = L_trans.T
    print("L")
    print(L)
    
    # 5. check new poles again
    new_A = A - L@C
    eigVal, eigVec = np.linalg.eig(new_A)
    print(f'eig-vals (controlled)')
    print(eigVal)