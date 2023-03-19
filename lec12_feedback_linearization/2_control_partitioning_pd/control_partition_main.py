from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint

class parameters:
    def __init__(self):
        # self.M =  np.array([[1, 0.1], [0.1, 2]])
        self.M = np.random.rand(2, 2)
        # 이거 항상 대각행렬임???
        # self.C =  np.array([[0.2, 0], [0, 0.1]])
        self.C = np.ones(2) * np.random.rand()
        self.K = np.array([[5, 1], [1, 10]])
        
        # Kp Kd 행렬은 항상 대각 행렬
        # q1의 상태와 q1의 상태가 독립적이라는 가정
        self.Kp = 100 * np.identity(2)
        # self.Kp = 100 * np.array([[1, 1], [1, 1]])
        self.Kd = 2 * np.sqrt(self.Kp)
        self.qdes = np.array([0.5, 0.1])

        # self.M_hat = self.M + np.array([[0.1, 0.01], [0.1, 0.01]])
        # self.C_hat = self.C + np.array([[0.2, 0.03], [0.3, 0.05]])
        # self.K_hat = self.K + np.array([[0.5, 0.05], [0.5, 0.05]])
        
        self.M_hat = self.M + 0.1 * np.random.rand(2, 2)
        self.C_hat = self.C + 0.1 * np.random.rand(2, 2)
        self.K_hat = self.K + 0.1 * np.random.rand(2, 2)

def control_partition_rhs(z,t, M,C,K,Kp,Kd,qdes,M_hat,C_hat,K_hat):

    q1, q1dot = z[0], z[1]
    q2, q2dot = z[2], z[3]
    
    q = np.array([q1,q2]);
    qdot = np.array([q1dot,q2dot]);

    # controller d
    # noise를 추가한 M_hat이 사용됨에 유의
    # tau = M.dot( -Kp.dot(q-qdes)-Kd.dot(qdot)) + C.dot(qdot) + K.dot(q)
    
    # imperfect model
    # K_hat.dot(q-qdes)도 해보자.
    tau = M_hat.dot( -Kp.dot(q-qdes)-Kd.dot(qdot)) + C_hat.dot(qdot) + K_hat.dot(q)

    # 여기에서는 M_hat이 아닌 M이 사용되었음
    A = M
    b = -C.dot(qdot)-K.dot(q)+tau
    invA = np.linalg.inv(A)
    qddot = invA.dot(b)

    zdot = np.array([qdot[0], qddot[0],qdot[1],qddot[1]]);

    return zdot

def plot(t, z, parms):
    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(t,z[:,0])
    plt.plot(t, parms.qdes[0]*np.ones(len(t)),'r+');
    plt.xlabel("t")
    plt.ylabel("q1")
    plt.title("Plot of position vs time")

    plt.subplot(2, 1, 2)
    plt.plot(t,z[:,2])
    plt.plot(t, parms.qdes[1]*np.ones(len(t)),'r+');
    plt.xlabel("t")
    plt.ylabel("q2")

    plt.show()

if __name__=="__main__":
    parms = parameters()
    
    q1, q1dot, q2, q2dot = 0, 0, 10, 0
    t0, tend = 0, 10
    
    M, C, K = parms.M, parms.C, parms.K
    Kp, Kd, qdes = parms.Kp, parms.Kd, parms.qdes
    M_hat, C_hat, K_hat = parms.M_hat, parms.C_hat, parms.K_hat
    

    t = np.linspace(t0, tend, 101)
    z0 = np.array([q1, q1dot, q2, q2dot])
    z = odeint(control_partition_rhs, z0, t, \
            args=(M,C,K, Kp,Kd,qdes, M_hat,C_hat,K_hat )
        )

    plot(t, z, parms)