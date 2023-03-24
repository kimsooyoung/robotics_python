from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.integrate import odeint

pi = np.pi

class Parameters():
    def __init__(self):
        self.m1 = 1
        self.m2 = 1
        self.l = 1
        self.c1 = self.l/2
        self.c2 = self.l/2
        
        self.I1 = self.m1*self.l**2/12
        self.I2 = self.m2*self.l**2/12
        
        self.g = 9.81
        
        self.kp1 = 100
        self.kd1 = 2*np.sqrt(self.kp1)
        self.kp2 = 100
        self.kd2 = 2*np.sqrt(self.kp2)
        
        
def link1_traj(ts):
    
    a0 =  -pi/2 - 0.5
    a1 =  0
    a2 =  0.333333333333333
    a3 =  -0.0740740740740741    
    
    pose = a0 + a1*ts + a2*ts**2 + a3*ts**3
    vel = a1 + 2*a2*ts + 3*a3*ts**2
    acc = 2*a2 + 6*a3*ts
    
    return pose, vel, acc

def link2_traj(ts):
    
    T1 = ts[:100]
    T2 = ts[100:]
    
    a10 =  0
    a11 =  0
    a12 =  0.666666666666667 - 0.666666666666667*pi
    a13 =  -0.296296296296296 + 0.296296296296296*pi
    pose1 = a10 + a11*T1 + a12*T1**2 + a13*T1**3
    vel1 = a11 + 2*a12*T1 + 3*a13*T1**2
    acc1 = 2*a12 + 6*a13*T1
    
    a20 =  -2.0 + 2.0*pi
    a21 =  4.0 - 4.0*pi
    a22 =  -2.0 + 2.0*pi
    a23 =  0.296296296296296 - 0.296296296296296*pi
    pose2 = a20 + a21*T2 + a22*T2**2 + a23*T2**3
    vel2 = a21 + 2*a22*T2 + 3*a23*T2**2
    acc2 = 2*a22 + 6*a23*T2
    
    pose = np.concatenate((pose1, pose2))
    vel  = np.concatenate((vel1, vel2))
    acc  = np.concatenate((acc1, acc2))
    
    return pose, vel, acc

def twolink_dynamics(t,z, args):
    
    theta1, omega1, theta2, omega2 = z

    M11 =  1.0*I1 + 1.0*I2 + c1**2*m1 + m2*(c2**2 + 2*c2*l*cos(theta2) + l**2)
    M12 =  1.0*I2 + c2*m2*(c2 + l*cos(theta2))
    M21 =  1.0*I2 + c2*m2*(c2 + l*cos(theta2))
    M22 =  1.0*I2 + c2**2*m2 

    C1 =  -c2*l*m2*theta2dot*(2.0*theta1dot + 1.0*theta2dot)*sin(theta2)
    C2 =  c2*l*m2*theta1dot**2*sin(theta2) 

    G1 =  g*(c1*m1*cos(theta1) + m2*(c2*cos(theta1 + theta2) + l*cos(theta1)))
    G2 =  c2*g*m2*cos(theta1 + theta2) 
    

if __name__=="__main__":
    
    params = Parameters()
    
    m1, m2, c1, c2, l = params.m1, params.m2, params.c1, params.c2, params.l
    I1, I2, g = params.I1, params.I2, params.g
    kp1, kd1, kp2, kd2 = params.kp1, params.kd1, params.kp2, params.kd2
    
    t0, t1, tend = 0, 1.5, 3.0
    ts = np.linspace(t0, tend, 200)
    
    # traj generation
    q1_p_ref, q1_v_ref, q1_a_ref = link1_traj(ts)
    q2_p_ref, q2_v_ref, q2_a_ref = link2_traj(ts)
    
    # initial conditions
    z0 = np.array([q1_p_ref[0], q1_v_ref[0], q2_p_ref[0], q2_v_ref[0]])
    
    z = np.zeros((len(ts), 4))
    z[0] = z0
    tau = np.zeros((len(ts), 2))
    tau[0] = np.array([0, 0])
    
    for i in range(len(q1_a_ref)-1):
        
        t_temp = np.array([ ts[i], ts[i+1] ])
        
        result = odeint(twolink_dynamics, z0, t_temp, args=args)
        