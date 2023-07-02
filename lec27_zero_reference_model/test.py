import cython_test 
import numpy as np

z = np.array([1.0, 2.0], dtype=np.float64)
params = np.array([0.1, 0.2], dtype=np.float64)

print(cython_test.test(z, 0.1, params))

import nlink_rhs_cython

# z: size 6 ndarray
# t: float64
# params: size 13 ndarray
z = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
t = 0.1
params = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, \
    0.7, 0.8, 0.9, 1.0, 1.1, 1.2, \
    1.3
], dtype=np.float64)
M, C, G = nlink_rhs_cython.nlink_rhs(z, t, params)

print(f"M: {M}")
print(f"C: {C}")
print(f"G: {G}")