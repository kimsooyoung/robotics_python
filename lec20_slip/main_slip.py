import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# TODO:
# 1. flight eom
#    - contact event
#    - apex event
# 2. stance eom
#    - release event

class Params:
    def __init__(self):
        self.g = 9.81
        self.ground = 0.0
        self.l = 1
        self.m = 1
        
        # sprint stiffness
        self.k = 100
        # fixed angle
        self.theta = 10 * np.pi / 180
        
        self.pause = 0.005
        self.fps = 10

def flight(t, z, g):
    x, x_dot, y, y_dot = z
    
    return [x_dot, 0, y_dot, -g]

def contact(t, z, l0, theta):
    x, x_dot, y, y_dot = z
    # contact event
    return y - l0 * np.cos(theta)
# contact.direction = -1
# contact.terminal = True

def apex(t, z):
    x, x_dot, y, y_dot = z
    return y_dot
# apex.direction = 0
# apex.terminal = True

def stance(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    
    l = np.sqrt(x**2 + y**2)
    F_spring = k * (l0 - l)
    Fx_spring = F_spring * x / l
    Fy_spring = F_spring * y / l
    Fy_gravity = m*g
    
    x_dd = (Fx_spring) / m
    y_dd = (Fy_spring - Fy_gravity) / m
    
    return [x_dot, x_dd, y_dot, y_dd]

def release(t, z, l0):
    x, x_dot, y, y_dot = z
    l = np.sqrt(x**2 + y**2)
    
    return l - l0
# release.direction = +1
# release.terminal = True

if __name__=="__main__":
    
    params = Params()
    
    