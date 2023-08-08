from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.integrate import odeint

class parameters:
    def __init__(self):
        self.m1 = 1
        self.m2 = 1
        self.k1 = 2
        self.k2 = 3

        self.pause = 0.01
        self.fps =20

def spring_mass_rhs(x,t,m1,m2,k1,k2):

    A = np.array([
                [0,0,1,0],
                [0,0,0,1],
                [-(k1/m1+k2/m1), k2/m1, 0, 0],
                [k2/m2, -k2/m2, 0, 0]
                ])


    C = np.array([
                 [0,0,1,0],
                 [0,0,0,1]
                 ])

    #pole placement (copy pasted from spring_mass_obsv.py)
    L =  [[-14.71334988, -14.60085979],
         [-14.78927377, -24.11788484],
         [ 11.73034952,  -0.40869829],
         [ -0.48187707,  11.26965048]]

    O_44 = np.zeros((4,4))
    LC = L@C
    A_LC = np.subtract(A, LC)

    # no observer
    # e1_d = A * e1
    # e2_d = A * e2
    Abig = np.block([ \
         [A,   O_44], \
        [O_44, A ] \
     ])

    # with Luenberg observer
    # e1_d = A * e1
    # e2_d = LC*e1 + (A-LC)*e2
    # e2_d = A * e2 + LC*(e1 - e2)
    # => 첫번째 L*C는 뭐지 ??
    
    # 만약 이렇게 하면 모든 값들이 0으로 가버린다.
    Abig = np.block([ \
         [A,   O_44], \
        [O_44, A_LC ] \
     ])
    
    # 지금 우리가 하고자 하는 것은 real value인 e1에 가깝도록 하는 것이기 때문에
    # e2_d = LC*e1 + (A-LC)*e2를 해주는 것이다.
    Abig = np.block([ \
         [A,   O_44], \
        [LC, A_LC ] \
     ])

    return Abig@x

def plot(t, x, parameters):

     plt.figure(1)
     
     plt.subplot(2,2,1)
     plt.plot(t,x[:,0],'r-.')
     plt.plot(t,x[:,4],'b');
     plt.ylabel("position q1")
     plt.legend(['act','est'])
     
     plt.subplot(2,2,3)
     plt.plot(t,x[:,1],'r-.')
     plt.plot(t,x[:,5],'b');
     plt.legend(['act','est'])
     plt.ylabel("position q2")
     plt.xlabel("time t")

     plt.subplot(2,2,2)
     plt.plot(t,x[:,2],'r-.')
     plt.plot(t,x[:,6],'b');
     plt.ylabel("velocity q1dot ")
     plt.legend(['act','est'])
     
     plt.subplot(2,2,4)
     plt.plot(t,x[:,3],'r-.')
     plt.plot(t,x[:,7],'b');
     plt.ylabel("velocity q2dot ")
     plt.xlabel("time t")
     plt.legend(['act','est'])

     plt.show(block=False)
     plt.pause(5)
     plt.close()

if __name__=='__main__':
     parms = parameters()

     x0 = np.array([0.5,0,0,0])
     x0est = np.array([0.2,0,0,0])
     x0big = np.concatenate((x0, x0est))

     t0, tend = 0, 5

     t = np.linspace(t0, tend, 101)
     x = odeint(spring_mass_rhs, x0big, t, args=(parms.m1,parms.m2,parms.k1,parms.k2))
     
     plot(t, x, parameters)