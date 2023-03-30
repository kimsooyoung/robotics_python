from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint
import control

class parameters:
    def __init__(self):
        self.m1 = 1
        self.m2 = 1
        self.k1 = 2
        self.k2 = 3

        self.A = np.array([
                    [0,0,1,0],
                    [0,0,0,1],
                    [-(self.k1/self.m1+self.k2/self.m1), self.k2/self.m1, 0, 0],
                    [self.k2/self.m2, -self.k2/self.m2, 0, 0]
                    ])

        self.C = np.array([
                     [0,0,1,0],
                     [0,0,0,1]
                     ])

        self.G = np.array([
                     [0,0],
                     [0,0],
                     [1,0],
                     [0,1]
                     ])

def spring_mass_rhs(x,t,A,C,G,L,f1_disturb,f2_disturb,q1dot_sensor,q2dot_sensor):

    O_44 = np.zeros((4,4))
    LC = np.matmul(L,C)
    A_LC = np.subtract(A,LC)

    O_42 = np.zeros((4,2))
    Gbig = np.block([ \
         [G], \
         [O_42 ] \
         ])
    # print("Gbig", Gbig)

    # process noise
    w = np.array([f1_disturb,f2_disturb])
    # print("w", w)

    Lbig = np.block([ \
         [O_42 ], \
         [L] \
         ])
    
    # measurement noise 
    v = np.array([q1dot_sensor,q2dot_sensor])

    #no observer
    # Abig = np.block([ \
    #      [A,   O_44], \
    #     [O_44, A ] \
    #      ])

    #with filter gain L
    Abig = np.block([ \
         [A,   O_44], \
        [LC, A_LC ] \
         ])

    # e1_d = A*e1 + G*w + L*v - L=0
    # e2_d = LC*e1 + (A-LC)*e1 + G*w + L*v - G=0
    # xdot = Abig.dot(x) + Gbig.dot(w) - Lbig.dot(v)
    xdot = Abig.dot(x) + Gbig.dot(w) + Lbig.dot(v)

    return xdot

def plot(t, xbig):
    
    plt.figure(1)
    
    plt.subplot(2,2,1)
    plt.plot(t,xbig[:,0],'r-.')
    plt.plot(t,xbig[:,4],'b');
    plt.ylabel("position q1")
    plt.legend(['act','est'])
    
    plt.subplot(2,2,3)
    plt.plot(t,xbig[:,1],'r-.')
    plt.plot(t,xbig[:,5],'b');
    plt.legend(['act','est'])
    plt.ylabel("position q2")
    plt.xlabel("time t")

    plt.subplot(2,2,2)
    plt.plot(t,xbig[:,2],'r-.')
    plt.plot(t,xbig[:,6],'b');
    plt.ylabel("velocity q1dot ")
    plt.legend(['act','est'])
    
    plt.subplot(2,2,4)
    plt.plot(t,xbig[:,3],'r-.')
    plt.plot(t,xbig[:,7],'b');
    plt.ylabel("velocity q2dot ")
    plt.xlabel("time t")
    plt.legend(['act','est'])

    plt.show(block=False)
    plt.pause(10)
    plt.close()

if __name__=="__main__":

    np.random.seed(1)
    parms = parameters()

    #These are true values of disturbance and noise statistics
    Qe_act = np.diag([2,3])
    Re_act = np.diag([0.5,0.5])
    f1_mean, f1_dev = 0, np.sqrt(Qe_act[0,0])
    f2_mean, f2_dev = 0, np.sqrt(Qe_act[1,1])
    q1dot_mean, q1dot_dev = 0, np.sqrt(Re_act[0,0])
    q2dot_mean, q2dot_dev = 0, np.sqrt(Re_act[1,1])

    A, C = parms.A, parms.C

    # #Design Luenberg observer (Deterministic system)
    # p = np.array([-5,-5.5,-6.5,-6])
    # L_trans = control.place(A.T , C.T, p)
    # L = np.transpose(L_trans)
    # print("(Luenberg gain) L = ",L)

    #Design Kalman filter (Stochastic system)
    Qe = Qe_act
    Re = Re_act
    G = parms.G
    print("Qe = ",Qe)
    print("Re = ",Re)
    
    L, P, E = control.lqe(A, G, C, Qe, Re)
    print("Kalman Gain: L = ",L)

    x0 = np.array([0.5,0,0,0])
    x0est = np.array([0.2,0,0,0])
    x0big = np.concatenate((x0, x0est))

    t0, tend = 0, 5

    Npts = 101
    N = 4
    shape = (Npts,2*N)
    
    t = np.linspace(t0, tend, 101)
    xbig = np.zeros((101, 8))
    xbig[0] = x0big

    for i in range(0, 101-1):
        f1_disturb = np.random.normal(f1_mean,f1_dev)
        f2_disturb = np.random.normal(f2_mean,f2_dev)
        q1dot_sensor = np.random.normal(q1dot_mean,q1dot_dev)
        q2dot_sensor = np.random.normal(q2dot_mean,q2dot_dev)
        
        t_temp = np.array([t[i], t[i+1]])
        physical_parms = (parms.A,parms.C,parms.G)
        control_parms = (L,f1_disturb,f2_disturb,q1dot_sensor,q2dot_sensor)
        all_parms = physical_parms + control_parms
        
        xbig_temp = odeint(spring_mass_rhs, x0big, t_temp, args=all_parms)
        
        x0big = xbig_temp[1]
        xbig[i+1] = x0big
    
    plot(t, xbig)