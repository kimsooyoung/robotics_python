import osqp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from spring import spring

import control
from scipy import sparse

A = np.array([
    [ 0. ,  1. ],
    [-0.1, -0.2]
])

B = np.array([
    [0.  ],
    [0.05]
])

x = np.array([
    [-0.11415525],
    [-0.11415525]
])
u = np.array([[-2.96803653]])

new_x = A@x + B@u

print(new_x)