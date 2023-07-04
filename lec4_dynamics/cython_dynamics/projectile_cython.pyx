import numpy as np 
from libc.math cimport sqrt

cimport numpy as cnp
cnp.import_array()

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def projectile(cnp.ndarray[DTYPE_t, ndim=1] z, t, float m, float g, float c):

    cdef DTYPE_t x    = z[0]
    cdef DTYPE_t xdot = z[1]
    cdef DTYPE_t y    = z[2]
    cdef DTYPE_t ydot = z[3]
    
    cdef DTYPE_t v, dragX, dragY, ax, ay

    cdef cnp.ndarray z_out = np.zeros([4], dtype=DTYPE)

    v = sqrt(xdot**2+ydot**2)

    #%%%% drag is prop to v^2
    dragX = c * v * xdot
    dragY = c * v * ydot

    #%%%% net acceleration 
    ax =  0 - (dragX / m) #xddot
    ay = -g - (dragY / m) #yddot

    z_out[0] = xdot
    z_out[1] = ax
    z_out[2] = ydot
    z_out[3] = ay

    return z_out