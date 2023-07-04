import numpy as np 
import numpy as np
from libc.math cimport sin, cos

cimport numpy as cnp
cnp.import_array()

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t


cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def collision(float t, cnp.ndarray[DTYPE_t, ndim=1] z_in, cnp.ndarray[DTYPE_t, ndim=1] params):
    
    cdef DTYPE_t x = z_in[0]
    cdef DTYPE_t xd = z_in[1]
    cdef DTYPE_t y = z_in[2]
    cdef DTYPE_t yd = z_in[3]
    cdef DTYPE_t z = z_in[4]
    cdef DTYPE_t zd = z_in[5]

    cdef DTYPE_t phi = z_in[6]
    cdef DTYPE_t phid = z_in[7]
    cdef DTYPE_t theta = z_in[8]
    cdef DTYPE_t thetad = z_in[9]
    cdef DTYPE_t psi = z_in[10]
    cdef DTYPE_t psid = z_in[11]

    cdef DTYPE_t phi_lh = z_in[12]
    cdef DTYPE_t phi_lhd = z_in[13]
    cdef DTYPE_t theta_lh = z_in[14]
    cdef DTYPE_t theta_lhd = z_in[15]
    cdef DTYPE_t psi_lh = z_in[16]
    cdef DTYPE_t psi_lhd = z_in[17]
    cdef DTYPE_t theta_lk = z_in[18]
    cdef DTYPE_t theta_lkd = z_in[19]

    cdef DTYPE_t phi_rh = z_in[20]
    cdef DTYPE_t phi_rhd = z_in[21]
    cdef DTYPE_t theta_rh = z_in[22]
    cdef DTYPE_t theta_rhd = z_in[23]
    cdef DTYPE_t psi_rh = z_in[24]
    cdef DTYPE_t psi_rhd = z_in[25]
    cdef DTYPE_t theta_rk = z_in[26]
    cdef DTYPE_t theta_rkd = z_in[27]

    cdef DTYPE_t l1 = params[0]
    cdef DTYPE_t l2 = params[1]
    cdef DTYPE_t w = params[2]

    cdef DTYPE_t gstop

    # x, xd, y, yd, z, zd, \
    #     phi, phid, theta, thetad, psi, psid, \
    #     phi_lh, phi_lhd, theta_lh, theta_lhd, \
    #     psi_lh, psi_lhd, theta_lk, theta_lkd, \
    #     phi_rh, phi_rhd, theta_rh, theta_rhd, \
    #     psi_rh, psi_rhd, theta_rk, theta_rkd = z
        
    # l1, l2, w = params.l1, params.l2, params.w
    
    gstop = 2*w*cos(psi)*sin(phi) + 2*w*cos(phi)*sin(psi)*sin(theta) - l1*cos(phi)*cos(phi_lh)*cos(theta)*cos(theta_lh) + l1*cos(phi)*cos(phi_rh)*cos(theta)*cos(theta_rh) + l1*cos(psi_lh)*sin(phi)*sin(psi)*sin(theta_lh) + l1*cos(psi)*sin(phi)*sin(psi_lh)*sin(theta_lh) - l1*cos(psi_rh)*sin(phi)*sin(psi)*sin(theta_rh) + l1*cos(psi)*sin(phi)*sin(psi_rh)*sin(theta_rh) + l1*cos(psi_lh)*cos(psi)*cos(theta_lh)*sin(phi)*sin(phi_lh) + l1*cos(psi_rh)*cos(psi)*cos(theta_rh)*sin(phi)*sin(phi_rh) - l1*cos(phi)*cos(psi_lh)*cos(psi)*sin(theta)*sin(theta_lh) + l1*cos(phi)*cos(psi_rh)*cos(psi)*sin(theta)*sin(theta_rh) + l2*cos(phi)*cos(phi_lh)*cos(theta)*sin(theta_lh)*sin(theta_lk) - l2*cos(phi)*cos(phi_rh)*cos(theta)*sin(theta_rh)*sin(theta_rk) + l2*cos(psi_lh)*cos(theta_lh)*sin(phi)*sin(psi)*sin(theta_lk) + l2*cos(psi_lh)*cos(theta_lk)*sin(phi)*sin(psi)*sin(theta_lh) + l2*cos(psi)*cos(theta_lh)*sin(phi)*sin(psi_lh)*sin(theta_lk) + l2*cos(psi)*cos(theta_lk)*sin(phi)*sin(psi_lh)*sin(theta_lh) - l2*cos(psi_rh)*cos(theta_rh)*sin(phi)*sin(psi)*sin(theta_rk) - l2*cos(psi_rh)*cos(theta_rk)*sin(phi)*sin(psi)*sin(theta_rh) + l2*cos(psi)*cos(theta_rh)*sin(phi)*sin(psi_rh)*sin(theta_rk) + l2*cos(psi)*cos(theta_rk)*sin(phi)*sin(psi_rh)*sin(theta_rh) - l1*cos(theta_lh)*sin(phi)*sin(phi_lh)*sin(psi_lh)*sin(psi) + l1*cos(theta_rh)*sin(phi)*sin(phi_rh)*sin(psi_rh)*sin(psi) + l1*cos(phi)*sin(psi_lh)*sin(psi)*sin(theta)*sin(theta_lh) + l1*cos(phi)*sin(psi_rh)*sin(psi)*sin(theta)*sin(theta_rh) - l2*cos(phi)*cos(phi_lh)*cos(theta)*cos(theta_lh)*cos(theta_lk) + l2*cos(phi)*cos(phi_rh)*cos(theta)*cos(theta_rh)*cos(theta_rk) - l2*cos(psi_lh)*cos(psi)*sin(phi)*sin(phi_lh)*sin(theta_lh)*sin(theta_lk) - l2*cos(theta_lh)*cos(theta_lk)*sin(phi)*sin(phi_lh)*sin(psi_lh)*sin(psi) - l2*cos(psi_rh)*cos(psi)*sin(phi)*sin(phi_rh)*sin(theta_rh)*sin(theta_rk) + l2*cos(theta_rh)*cos(theta_rk)*sin(phi)*sin(phi_rh)*sin(psi_rh)*sin(psi) + l2*cos(phi)*cos(theta_lh)*sin(psi_lh)*sin(psi)*sin(theta)*sin(theta_lk) + l2*cos(phi)*cos(theta_lk)*sin(psi_lh)*sin(psi)*sin(theta)*sin(theta_lh) + l2*cos(phi)*cos(theta_rh)*sin(psi_rh)*sin(psi)*sin(theta)*sin(theta_rk) + l2*cos(phi)*cos(theta_rk)*sin(psi_rh)*sin(psi)*sin(theta)*sin(theta_rh) + l2*sin(phi)*sin(phi_lh)*sin(psi_lh)*sin(psi)*sin(theta_lh)*sin(theta_lk) - l2*sin(phi)*sin(phi_rh)*sin(psi_rh)*sin(psi)*sin(theta_rh)*sin(theta_rk) + l2*cos(psi_lh)*cos(psi)*cos(theta_lh)*cos(theta_lk)*sin(phi)*sin(phi_lh) + l2*cos(psi_rh)*cos(psi)*cos(theta_rh)*cos(theta_rk)*sin(phi)*sin(phi_rh) - l2*cos(phi)*cos(psi_lh)*cos(psi)*cos(theta_lh)*sin(theta)*sin(theta_lk) - l2*cos(phi)*cos(psi_lh)*cos(psi)*cos(theta_lk)*sin(theta)*sin(theta_lh) + l2*cos(phi)*cos(psi_rh)*cos(psi)*cos(theta_rh)*sin(theta)*sin(theta_rk) + l2*cos(phi)*cos(psi_rh)*cos(psi)*cos(theta_rk)*sin(theta)*sin(theta_rh) + l1*cos(phi)*cos(psi_lh)*cos(theta_lh)*sin(phi_lh)*sin(psi)*sin(theta) + l1*cos(phi)*cos(psi)*cos(theta_lh)*sin(phi_lh)*sin(psi_lh)*sin(theta) + l1*cos(phi)*cos(psi_rh)*cos(theta_rh)*sin(phi_rh)*sin(psi)*sin(theta) - l1*cos(phi)*cos(psi)*cos(theta_rh)*sin(phi_rh)*sin(psi_rh)*sin(theta) + l2*cos(phi)*cos(psi_lh)*cos(theta_lh)*cos(theta_lk)*sin(phi_lh)*sin(psi)*sin(theta) + l2*cos(phi)*cos(psi)*cos(theta_lh)*cos(theta_lk)*sin(phi_lh)*sin(psi_lh)*sin(theta) + l2*cos(phi)*cos(psi_rh)*cos(theta_rh)*cos(theta_rk)*sin(phi_rh)*sin(psi)*sin(theta) - l2*cos(phi)*cos(psi)*cos(theta_rh)*cos(theta_rk)*sin(phi_rh)*sin(psi_rh)*sin(theta) - l2*cos(phi)*cos(psi_lh)*sin(phi_lh)*sin(psi)*sin(theta)*sin(theta_lh)*sin(theta_lk) - l2*cos(phi)*cos(psi)*sin(phi_lh)*sin(psi_lh)*sin(theta)*sin(theta_lh)*sin(theta_lk) - l2*cos(phi)*cos(psi_rh)*sin(phi_rh)*sin(psi)*sin(theta)*sin(theta_rh)*sin(theta_rk) + l2*cos(phi)*cos(psi)*sin(phi_rh)*sin(psi_rh)*sin(theta)*sin(theta_rh)*sin(theta_rk)
    
    return gstop
