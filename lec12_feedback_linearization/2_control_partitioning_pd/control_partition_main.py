from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint

class parameters:
    def __init__(self):
        self.M =  np.array([[1, 0.1], [0.1, 2]])
        self.C =  np.array([[0.2, 0], [0, 0.1]])
        self.K = np.array([[5, 1], [1, 10]])
        self.Kp = 10*np.identity(2);
        self.Kd = 2*np.sqrt(self.Kp);
        self.qdes = np.array([0.5, 0.1])

        self.M_hat = self.M + np.array([[0.1, 0.01], [0.1, 0.01]])
        self.C_hat = self.C + np.array([[0.2, 0.03], [0.3, 0.05]])
        self.K_hat = self.K + np.array([[0.5, 0.05], [0.5, 0.05]])

def control_partition_rhs(z,t,M,C,K,Kp,Kd,qdes,M_hat,C_hat,K_hat):

    q1 = z[0];
    q1dot = z[1];
    q2 = z[2];
    q2dot = z[3];

    q = np.array([q1,q2]);
    qdot = np.array([q1dot,q2dot]);

    #controller d
    # tau = M.dot( -Kp.dot(q-qdes)-Kd.dot(qdot)) + C.dot(qdot) + K.dot(q)
    tau = M_hat.dot( -Kp.dot(q-qdes)-Kd.dot(qdot)) + C_hat.dot(qdot) + K_hat.dot(q) #imperfect mode


    A = M ;
    b = -C.dot(qdot)-K.dot(q)+tau
    invA = np.linalg.inv(A)
    qddot = invA.dot(b)

    zdot = np.array([qdot[0], qddot[0],qdot[1],qddot[1]]);

    return zdot

parms = parameters()
q1 = 0;
q1dot = 0;
q2 = 0;
q2dot = 0;
t0 = 0;
tend = 10;

t = np.linspace(t0, tend, 101)
z0 = np.array([q1, q1dot, q2, q2dot])
z = odeint(control_partition_rhs, z0, t, \
          args=(parms.M,parms.C,parms.K,parms.Kp,parms.Kd,parms.qdes,parms.M_hat,parms.C_hat,parms.K_hat))



plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(t,z[:,0])
plt.plot(t,parms.qdes[0]*np.ones(len(t)),'r+');
plt.xlabel("t")
plt.ylabel("q1")
plt.title("Plot of position vs time")

plt.subplot(2, 1, 2)
plt.plot(t,z[:,2])
plt.plot(t,parms.qdes[1]*np.ones(len(t)),'r+');
plt.xlabel("t")
plt.ylabel("q2")

plt.show()
# plt.show(block=False)
# plt.pause(2)
# plt.close()
