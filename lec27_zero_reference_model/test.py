import cython_test 
import numpy as np

z = np.array([1.0, 2.0], dtype=np.float64)
params = np.array([0.1, 0.2], dtype=np.float64)

print(cython_test.test(z, 0.1, params))