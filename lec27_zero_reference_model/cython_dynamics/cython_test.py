import cython_test 
import numpy as np

z = np.array([1.0, 2.0], dtype=np.float64)
params = np.array([0.1, 0.2], dtype=np.float64)

print(cython_test.test(z, 0.1, params))

import nlink_rhs_cython
import nlink_rhs

def cython_main():
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
    M, C, G = nlink_rhs_cython.nlink_rhs(z, params)
    
    return M, C, G

def pure_python_main():
    z = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    t = 0.1
    params = np.array([
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, \
        0.7, 0.8, 0.9, 1.0, 1.1, 1.2, \
        1.3
    ], dtype=np.float64)
    M, C, G = nlink_rhs.nlink_rhs(z, params)
    
    return M, C, G

# value check
M, C, G = cython_main()
print(f"M: {M}")
print(f"C: {C}")
print(f"G: {G}")

# time check 
import timeit
print(f"cython: {timeit.timeit(cython_main, number=100)}")
print(f"python: {timeit.timeit(pure_python_main, number=100)}")
