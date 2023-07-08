import time
import numpy as np 
from controller import controller


class Parameters:
    
    def __init__(self):
        
        self.dof = 14
        
        # leg lengths, body width
        self.l0 = 1; self.l1 = 0.5; self.l2 = 0.5
        self.w = 0.1
        
        # mass and inertia
        self.mb = 70; self.mt = 10; self.mc = 5
        self.Ibx = 5; self.Iby = 3; self.Ibz = 2
        self.Itx = 1; self.Ity = 0.3; self.Itz = 2
        self.Icx = 0.5; self.Icy = 0.15; self.Icz = 1
        
        self.g = 9.8

        # total weight and height
        self.M = self.mb + 2*self.mt + 2*self.mc
        self.L = self.l1 + self.l2
        
        # misc other params
        self.stepAngle = 0.375 # #step length
        self.Impulse = 0.18 * self.M * np.sqrt(self.g * self.L) #push-off impulse
        self.kneeAngle = -1 #Knee bending to avoid scuffing
        self.Kp = 100 #gain for partial feedback linearization
        self.Kd = np.sqrt(self.Kp) 
        
        self.stance_foot_init = 'right'
        self.stance_foot = self.stance_foot_init
        
        self.fps = 10
        self.pause = 0.01


if __name__=="__main__":
    params = Parameters()
    z0 = np.array([
        -0.000049815000110,
        1.014525925343273,
        0.070715370139685,
        -0.001725772505528,
        0.999571112013753,
        0.000000000000035,
        0,
        0,
        0,
        0.000000000000001,
        -0.000000000000006,
        -0.000000000000003,
        0.000000000000006,
        0.000000000000043,
        -0.000000000000002,
        0.000000000000070,
        -0.000000000000013,
        -0.000000000000057,
        -0.999999999999981,
        0.000000000000061,
        -0.029247773468329,
        0.054577886084082,
        -0.001551040824746,
        -1.029460778006675,
        0.054690088280553,
        0.559306810759302,
        -0.000000000000001,
        -0.000000000000031
    ])
    t = 0.0
    
    params.t0 = 0.0
    params.tf = 0.2
    params.s0 = np.array([
        0,0,-0.000000000000006,0.000000000000006,-0.000000000000002,-0.000000000000013,-0.999999999999981,-0.000000000000001  
    ])
    params.sf = np.array([
        0,0,0,0,0.375000000000000,0,0,0
    ])
    params.v0 = 1.0e-13 * np.array([
        0,0.010000000000000,-0.030000000000000,0.430000000000000,0.700000000000000,-0.570000000000000,0.610000000000000,-0.310000000000000
    ])
    params.vf = np.array([
        0,0,0,0,0,0,0,0
    ])
    params.a0 = np.array([
        0,0,0,0,0,0,0,0
    ])    
    params.af = np.array([
        0,0,0,0,0,0,0,0
    ])
    
    tau = controller(z0, t, params)
    
    print(tau)
    
    # [[  2.03238413]
    # [ 70.14164182]
    # [ -0.37666925]
    # [ 36.57043994]
    # [ 60.17626552]
    # [-66.352943  ]
    # [-14.02914473]
    # [-33.16790982]]