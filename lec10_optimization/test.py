from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import interpolate
from scipy.integrate import odeint
import random

import scipy.optimize as opt


t = np.array([0, 0.1])
t_opt = t
u_opt = np.array([-2,2])


f = interpolate.interp1d(t_opt, u_opt)
u = f(t)

print(u)
# [-2.  2.]