from matplotlib import pyplot as plt
from scipy.integrate import odeint
import numpy as np

class Parameter():
    
    def __init__(self):
        self.M = np.random.rand(2, 2)
        self.C = np.ones(2) * np.random.rand()
        self.K = np.random.rand(2, 2)
        
        self.Kp = 100 * np.identity(2)
        self.Kd = 2 * np.sqrt(self.Kp)
        
        self.q_des = np.array([0.5, 1.0])
        
        self.M_hat = self.M + 0.1 * np.random.rand(2, 2)
        self.C_hat = self.C + 0.1 * np.random.rand(2, 2)
        self.K_hat = self.K + 0.1 * np.random.rand(2, 2)
        
        
def control_partition_rhs(z,t, M,C,K,Kp,Kd,q_des,M_hat,C_hat,K_hat):
    
    q = np.array([z[0], z[2]])
    q_dot = np.array([z[1], z[3]])
    
    tau = M_hat@(-Kp@(q-q_des)-Kd@q_dot) + C_hat@(q_dot) + K_hat@(q)
    
    A = M
    b = tau - ( C@q_dot + K@q )
    q_ddot = np.linalg.inv(A) @ b
    
    return [z[1], q_ddot[0], z[3], q_ddot[1]]
    

def plot(t, z, params):
    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(t, z[:,0])
    plt.plot(t, params.q_des[0] * np.ones(len(t)),'r+');
    plt.xlabel("t")
    plt.ylabel("q1")
    plt.title("Plot of position vs time")

    plt.subplot(2, 1, 2)
    plt.plot(t, z[:,2])
    plt.plot(t, params.q_des[1] * np.ones(len(t)),'r+');
    plt.xlabel("t")
    plt.ylabel("q2")
    
    plt.show()
    
if __name__=="__main__":
    params = Parameter()
    q1, q1_dot = 0, 0
    q2, q2_dot = 0, 0
    
    t0, t_end = 0, 10
    
    t = np.linspace(t0, t_end, 101)
    z0 = np.array([q1, q1_dot, q2, q2_dot])
    
    M, C, K = params.M, params.C, params.K
    M_hat, C_hat, K_hat = params.M_hat, params.C_hat, params.K_hat
    Kp, Kd = params.Kp, params.Kd
    q_des = params.q_des
    
    z = odeint(
        control_partition_rhs, z0, t, 
        args=(M, C, K, Kp, Kd, q_des, M_hat, C_hat, K_hat)
    )
    plot(t, z, params)