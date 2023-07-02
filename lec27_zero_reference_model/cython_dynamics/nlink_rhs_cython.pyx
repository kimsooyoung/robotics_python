import numpy as np 
from libc.math cimport sin, cos

cimport numpy as cnp
cnp.import_array()

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

# z: size 6 ndarray
# params: size 13 ndarray

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def nlink_rhs(cnp.ndarray[DTYPE_t, ndim=1] z, cnp.ndarray[DTYPE_t, ndim=1] params): 

    cdef DTYPE_t q_0 = z[0]
    cdef DTYPE_t u_0 = z[1]
    cdef DTYPE_t q_1 = z[2]
    cdef DTYPE_t u_1 = z[3]
    cdef DTYPE_t q_2 = z[4]
    cdef DTYPE_t u_2 = z[5]
    
    cdef DTYPE_t m_0 = params[0]
    cdef DTYPE_t I_0 = params[1]
    cdef DTYPE_t c_0 = params[2]
    cdef DTYPE_t l_0 = params[3]
    cdef DTYPE_t m_1 = params[4]
    cdef DTYPE_t I_1 = params[5]
    cdef DTYPE_t c_1 = params[6]
    cdef DTYPE_t l_1 = params[7]
    cdef DTYPE_t m_2 = params[8]
    cdef DTYPE_t I_2 = params[9]
    cdef DTYPE_t c_2 = params[10]
    cdef DTYPE_t l_2 = params[11]
    cdef DTYPE_t g = params[12]

    cdef cnp.ndarray[DTYPE_t, ndim=2] M = np.zeros([3, 3], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=2] C = np.zeros([3, 1], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=2] G = np.zeros([3, 1], dtype=DTYPE)

    cdef DTYPE_t M11, M12, M13, M21, M22, M23, M31, M32, M33
    cdef DTYPE_t C1, C2, C3
    cdef DTYPE_t G1, G2, G3

    M11 = 1.0*I_0 + 1.0*I_1 + 1.0*I_2 + c_0**2*m_0 + m_1*(c_1**2 + 2*c_1*l_0*cos(q_1) + l_0**2) + m_2*(c_2**2 + 2*c_2*l_0*cos(q_1 + q_2) + 2*c_2*l_1*cos(q_2) + l_0**2 + 2*l_0*l_1*cos(q_1) + l_1**2) 

    M12 = 1.0*I_1 + 1.0*I_2 + c_1*m_1*(c_1 + l_0*cos(q_1)) + m_2*(c_2**2 + c_2*l_0*cos(q_1 + q_2) + 2*c_2*l_1*cos(q_2) + l_0*l_1*cos(q_1) + l_1**2) 

    M13 = 1.0*I_2 + c_2*m_2*(c_2 + l_0*cos(q_1 + q_2) + l_1*cos(q_2)) 

    M21 = 1.0*I_1 + 1.0*I_2 + c_1*m_1*(c_1 + l_0*cos(q_1)) + m_2*(c_2**2 + c_2*l_0*cos(q_1 + q_2) + 2*c_2*l_1*cos(q_2) + l_0*l_1*cos(q_1) + l_1**2) 

    M22 = 1.0*I_1 + 1.0*I_2 + c_1**2*m_1 + m_2*(c_2**2 + 2*c_2*l_1*cos(q_2) + l_1**2) 

    M23 = 1.0*I_2 + c_2*m_2*(c_2 + l_1*cos(q_2)) 

    M31 = 1.0*I_2 + c_2*m_2*(c_2 + l_0*cos(q_1 + q_2) + l_1*cos(q_2)) 

    M32 = 1.0*I_2 + c_2*m_2*(c_2 + l_1*cos(q_2)) 

    M33 = 1.0*I_2 + c_2**2*m_2 


    C1 = -2.0*c_1*l_0*m_1*u_0*u_1*sin(q_1) - 1.0*c_1*l_0*m_1*u_1**2*sin(q_1) - 2.0*c_2*l_0*m_2*u_0*u_1*sin(q_1 + q_2) - 2.0*c_2*l_0*m_2*u_0*u_2*sin(q_1 + q_2) - 1.0*c_2*l_0*m_2*u_1**2*sin(q_1 + q_2) - 2.0*c_2*l_0*m_2*u_1*u_2*sin(q_1 + q_2) - 1.0*c_2*l_0*m_2*u_2**2*sin(q_1 + q_2) - 2.0*c_2*l_1*m_2*u_0*u_2*sin(q_2) - 2.0*c_2*l_1*m_2*u_1*u_2*sin(q_2) - 1.0*c_2*l_1*m_2*u_2**2*sin(q_2) - 2.0*l_0*l_1*m_2*u_0*u_1*sin(q_1) - 1.0*l_0*l_1*m_2*u_1**2*sin(q_1) 

    C2 = 1.0*c_1*l_0*m_1*u_0**2*sin(q_1) + 1.0*c_2*l_0*m_2*u_0**2*sin(q_1 + q_2) - 2.0*c_2*l_1*m_2*u_0*u_2*sin(q_2) - 2.0*c_2*l_1*m_2*u_1*u_2*sin(q_2) - 1.0*c_2*l_1*m_2*u_2**2*sin(q_2) + 1.0*l_0*l_1*m_2*u_0**2*sin(q_1) 

    C3 = c_2*m_2*(l_0*u_0**2*sin(q_1 + q_2) + l_1*u_0**2*sin(q_2) + 2*l_1*u_0*u_1*sin(q_2) + l_1*u_1**2*sin(q_2)) 


    G1 = g*(c_0*m_0*sin(q_0) + m_1*(c_1*sin(q_0 + q_1) + l_0*sin(q_0)) + m_2*(c_2*sin(q_0 + q_1 + q_2) + l_0*sin(q_0) + l_1*sin(q_0 + q_1))) 

    G2 = g*(c_1*m_1*sin(q_0 + q_1) + m_2*(c_2*sin(q_0 + q_1 + q_2) + l_1*sin(q_0 + q_1))) 

    G3 = c_2*g*m_2*sin(q_0 + q_1 + q_2) 

    M[0][0] = M11
    M[0][1] = M12
    M[0][2] = M13
    M[1][0] = M21
    M[1][1] = M22
    M[1][2] = M23
    M[2][0] = M31
    M[2][1] = M32
    M[2][2] = M33
    
    C[0][0] = C1
    C[1][0] = C2
    C[2][0] = C3

    G[0][0] = G1
    G[1][0] = G2
    G[2][0] = G3

    return M, C, G
