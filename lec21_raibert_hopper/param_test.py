
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

class Params:
    def __init__(self):
        self.g = 9.81
        self.ground = 0.0
        self.l = 1
        self.m = 1
        
        # sprint stiffness
        self.k = 200
        # fixed angle
        self.theta = 5 * (np.pi / 180)
        
        self.pause = 0.1
        self.fps = 10
        
        # new params for raibert hopper
        self.T = np.pi * np.sqrt(self.m/self.k)
        self.Kp = 0.1
        self.vdes = [0, 0.5, 0.6, 0.9, 1.0, 1.1, 0.7, 0.3, 0]
        

param = Params()
print(param.g)
param.g = 1000 
print(param.g)

angle1 = 1.3
print(np.arcsin(np.sin(angle1)))
