import scipy.optimize as opt
import numpy as np

inf = np.inf

def cost(param):
    x1, x2, x3, x4, x5 = param
    return x1**2 + x2**2 + x3**2 + x4**2 + x5**2

limits = opt.Bounds(
    [0.3, -inf, -inf, -inf, -inf],
    [inf,  inf,    5,  inf,  inf]
)

ineq_const = {
    'type': 'ineq',
    'fun': lambda x: np.array([ 5 - x[3]**2 - x[4]** 2 ]),
}

eq_const = {
    'type': 'eq',
    'fun': lambda x: np.array([ 
        5 - x[0] - x[1] - x[2],
        2 - x[2]**2 - x[3]
    ]),
}

x0 = np.array([1, 1, 1, 1, 1])
res = opt.minimize(cost, x0, method='SLSQP', 
               constraints=[eq_const, ineq_const], 
               options={'ftol': 1e-9, 'disp': True}, bounds=limits)
print(res.x)