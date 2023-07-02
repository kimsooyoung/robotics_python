import numpy as np 
import numpy as np
from libc.math cimport sin, cos

cimport numpy as cnp
cnp.import_array()

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

# TODO
# z, t type 
# params parse
# calculation

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def test(cnp.ndarray[DTYPE_t, ndim=1] z, float t, cnp.ndarray[DTYPE_t, ndim=1] params):

    cdef int vmax = z.shape[0]

    cdef float val1 = z[0]
    cdef float val2 = z[1]
    cdef float param1 = params[0]
    cdef float param2 = params[1]

    cdef cnp.ndarray[DTYPE_t, ndim=1] h = np.zeros([vmax], dtype=DTYPE)

    h[0] = val1 + param1
    h[1] = val2 + param2

    return h
