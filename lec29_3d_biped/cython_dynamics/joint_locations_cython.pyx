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
def joint_locations(float l0, float l1, float l2, float phi, float phi_lh, float phi_rh, float psi_lh, float psi_rh, float psi, float theta, float theta_lh, float theta_lk, float theta_rh, float theta_rk, float w, float x, float y, float z):
    
    cdef cnp.ndarray[DTYPE_t, ndim=1] B = np.zeros([3], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] H = np.zeros([4], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] LH = np.zeros([4], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] LK = np.zeros([4], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] LA = np.zeros([4], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] RH = np.zeros([4], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] RK = np.zeros([4], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] RA = np.zeros([4], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] b = np.zeros([4], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] rt = np.zeros([4], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] rc = np.zeros([4], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] lt = np.zeros([4], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] lc = np.zeros([4], dtype=DTYPE)

    cdef float t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19
    cdef float t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t31, t34, t35, t36, t37, t38, t39
    cdef float t40, t41, t42, t43, t44, t45, t46, t47, t48, t49, t54, t55, t56, t57, t58, t59, t60
    cdef float t61, t62, t63, t64, t65, t68, t69, t70, t71, t72, t73, t75, t76, t80, t81, t82, t83
    cdef float t84, t85, t86, t87, t88, t89, t90, t91, t92, t93, t94, t95, t96, t97, t98, t99, t100
    cdef float t101, t102, t103

    cdef float mt1, mt2, mt3, mt4, mt5, mt6
    cdef float et1, et2, et3, et4, et5, et6, et7, et8

    B[0] = x
    B[1] = y
    B[2] = z
    
    t2 = cos(phi)
    t3 = cos(phi_lh)
    t4 = cos(phi_rh)
    t5 = cos(psi_lh)
    t6 = cos(psi_rh)
    t7 = cos(psi)
    t8 = cos(theta)
    t9 = cos(theta_lh)
    t10 = cos(theta_lk)
    t11 = cos(theta_rh)
    t12 = cos(theta_rk)
    t13 = sin(phi)
    t14 = sin(phi_lh)
    t15 = sin(phi_rh)
    t16 = sin(psi_lh)
    t17 = sin(psi_rh)
    t18 = sin(psi)
    t19 = sin(theta)

    H[0] = x + l0*t19
    H[1] = y - l0*t8*t13
    H[2] = z + l0*t2*t8
    H[3] = 1.0

    t20 = sin(theta_lh)
    t21 = sin(theta_lk)
    t22 = sin(theta_rh)
    t23 = sin(theta_rk)
    t24 = t2*t7*w
    t25 = t7*t13*w
    t26 = t8*t18*w
    t27 = l1*t3*t9*t19
    t28 = l1*t4*t11*t19
    t29 = t2*t18*t19*w
    t31 = t13*t18*t19*w
    t34 = l1*t2*t3*t8*t9
    t35 = l1*t2*t4*t8*t11
    t36 = l1*t3*t8*t9*t13
    
    t37 = l1*t4*t8*t11*t13
    t38 = l1*t5*t7*t8*t20
    t39 = l1*t6*t7*t8*t22
    t40 = l1*t2*t5*t18*t20
    t41 = l1*t2*t7*t16*t20
    t42 = l1*t2*t6*t18*t22
    t43 = l1*t2*t7*t17*t22
    t44 = l1*t5*t13*t18*t20
    t45 = l1*t7*t13*t16*t20
    t46 = l1*t6*t13*t18*t22
    t47 = l1*t7*t13*t17*t22
    t48 = l1*t8*t16*t18*t20
    t49 = l1*t8*t17*t18*t22
    t54 = l1*t2*t5*t7*t9*t14
    t55 = l1*t2*t6*t7*t11*t15
    t56 = l1*t5*t7*t9*t13*t14
    t57 = l1*t6*t7*t11*t13*t15
    t58 = l1*t2*t5*t7*t19*t20
    t59 = l1*t2*t6*t7*t19*t22
    t60 = l1*t5*t8*t9*t14*t18
    t61 = l1*t7*t8*t9*t14*t16
    t62 = l1*t6*t8*t11*t15*t18
    t63 = l1*t7*t8*t11*t15*t17
    t64 = l1*t2*t9*t14*t16*t18
    t65 = l1*t2*t11*t15*t17*t18
    t68 = l1*t5*t7*t13*t19*t20
    t69 = l1*t6*t7*t13*t19*t22
    t70 = l1*t9*t13*t14*t16*t18
    t71 = l1*t11*t13*t15*t17*t18
    t72 = l1*t2*t16*t18*t19*t20
    t73 = l1*t2*t17*t18*t19*t22
    t75 = l1*t13*t16*t18*t19*t20
    t76 = l1*t13*t17*t18*t19*t22
    t80 = l1*t5*t9*t13*t14*t18*t19
    t81 = l1*t7*t9*t13*t14*t16*t19
    t82 = l1*t6*t11*t13*t15*t18*t19
    t83 = l1*t7*t11*t13*t15*t17*t19
    t96 = l1*t2*t5*t9*t14*t18*t19
    t97 = l1*t2*t7*t9*t14*t16*t19
    t98 = l1*t2*t6*t11*t15*t18*t19
    t99 = l1*t2*t7*t11*t15*t17*t19
    t30 = -t24
    t32 = -t25
    t33 = -t26
    t50 = -t27
    t51 = -t28
    t52 = -t29
    t53 = -t31

    LH[0] = t33 + x
    LH[1] = t24 + t53 + y
    LH[2] = t25 + t29 + z
    LH[3] = 1.0

    t66 = -t34
    t67 = -t35
    t74 = -t43
    t77 = -t47
    t78 = -t48 
    t79 = -t55
    t84 = -t57
    t85 = -t58
    t86 = -t59
    t87 = -t60
    t88 = -t61
    t89 = -t63
    t90 = -t64
    t91 = -t65
    t92 = -t70
    t93 = -t71
    t94 = -t73
    t95 = -t75
    t100 = -t80
    t101 = -t81

    LK[0] = t33+t38+t50+t78+t87+t88+x
    LK[1] = t24+t36+t40+t41+t53+t54+t68+t90+t95+t100+t101+y
    LK[2] = t25+t29+t44+t45+t56+t66+t72+t85+t92+t96+t97+z
    LK[3] = 1.0
    
    mt1 = t33+t38+t50+t78+t87+t88+x-l2*t3*t9*t10*t19+l2*t3*t19*t20*t21+l2*t5*t7*t8*t9*t21+l2*t5*t7*t8*t10*t20-l2*t8*t9*t16*t18*t21-l2*t8*t10*t16*t18*t20-l2*t5*t8*t9*t10*t14*t18-l2*t7*t8*t9*t10*t14*t16+l2*t5*t8*t14*t18*t20*t21+l2*t7*t8*t14*t16*t20*t21
    mt2 = t24+t36+t40+t41+t53+t54+t68+t90+t95+t100+t101+y+l2*t3*t8*t9*t10*t13+l2*t2*t5*t9*t18*t21+l2*t2*t5*t10*t18*t20+l2*t2*t7*t9*t16*t21+l2*t2*t7*t10*t16*t20-l2*t3*t8*t13*t20*t21+l2*t2*t5*t7*t9*t10*t14-l2*t2*t5*t7*t14*t20*t21-l2*t2*t9*t10*t14*t16*t18+l2*t5*t7*t9*t13*t19*t21+l2*t5*t7*t10*t13*t19*t20+l2*t2*t14*t16*t18*t20*t21-l2*t9*t13*t16*t18*t19*t21-l2*t10*t13*t16*t18*t19*t20-l2*t5*t9*t10*t13*t14*t18*t19-l2*t7*t9*t10*t13*t14*t16*t19+l2*t5*t13*t14*t18*t19*t20*t21+l2*t7*t13*t14*t16*t19*t20*t21
    mt3 = t25+t29+t44+t45+t56+t66+t72+t85+t92+t96+t97+z-l2*t2*t3*t8*t9*t10+l2*t2*t3*t8*t20*t21+l2*t5*t9*t13*t18*t21+l2*t5*t10*t13*t18*t20+l2*t7*t9*t13*t16*t21+l2*t7*t10*t13*t16*t20+l2*t5*t7*t9*t10*t13*t14-l2*t2*t5*t7*t9*t19*t21-l2*t2*t5*t7*t10*t19*t20-l2*t5*t7*t13*t14*t20*t21-l2*t9*t10*t13*t14*t16*t18+l2*t2*t9*t16*t18*t19*t21+l2*t2*t10*t16*t18*t19*t20+l2*t13*t14*t16*t18*t20*t21+l2*t2*t5*t9*t10*t14*t18*t19+l2*t2*t7*t9*t10*t14*t16*t19-l2*t2*t5*t14*t18*t19*t20*t21-l2*t2*t7*t14*t16*t19*t20*t21
    
    LA[0] = mt1
    LA[1] = mt2
    LA[2] = mt3
    LA[3] = 1.0

    RH[0] = t26+x
    RH[1] = t30+t31+y
    RH[2] = t32+t52+z
    RH[3] = 1.0
    
    t102 = -t83
    t103 = -t98

    RK[0] = t26+t39+t49+t51+t62+t89+x
    RK[1] = t30+t31+t37+t42+t69+t74+t76+t79+t82+t91+t102+y
    RK[2] = t32+t46+t52+t67+t77+t84+t86+t93+t94+t99+t103+z
    RK[3] = 1.0
    
    mt4 = t26+t39+t49+t51+t62+t89+x-l2*t4*t11*t12*t19+l2*t4*t19*t22*t23+l2*t6*t7*t8*t11*t23+l2*t6*t7*t8*t12*t22+l2*t8*t11*t17*t18*t23+l2*t8*t12*t17*t18*t22+l2*t6*t8*t11*t12*t15*t18-l2*t7*t8*t11*t12*t15*t17-l2*t6*t8*t15*t18*t22*t23+l2*t7*t8*t15*t17*t22*t23
    mt5 = t30+t31+t37+t42+t69+t74+t76+t79+t82+t91+t102+y+l2*t4*t8*t11*t12*t13+l2*t2*t6*t11*t18*t23+l2*t2*t6*t12*t18*t22-l2*t2*t7*t11*t17*t23-l2*t2*t7*t12*t17*t22-l2*t4*t8*t13*t22*t23-l2*t2*t6*t7*t11*t12*t15+l2*t2*t6*t7*t15*t22*t23-l2*t2*t11*t12*t15*t17*t18+l2*t6*t7*t11*t13*t19*t23+l2*t6*t7*t12*t13*t19*t22+l2*t2*t15*t17*t18*t22*t23+l2*t11*t13*t17*t18*t19*t23+l2*t12*t13*t17*t18*t19*t22+l2*t6*t11*t12*t13*t15*t18*t19-l2*t7*t11*t12*t13*t15*t17*t19-l2*t6*t13*t15*t18*t19*t22*t23+l2*t7*t13*t15*t17*t19*t22*t23
    mt6 = t32+t46+t52+t67+t77+t84+t86+t93+t94+t99+t103+z-l2*t2*t4*t8*t11*t12+l2*t2*t4*t8*t22*t23+l2*t6*t11*t13*t18*t23+l2*t6*t12*t13*t18*t22-l2*t7*t11*t13*t17*t23-l2*t7*t12*t13*t17*t22-l2*t6*t7*t11*t12*t13*t15-l2*t2*t6*t7*t11*t19*t23-l2*t2*t6*t7*t12*t19*t22+l2*t6*t7*t13*t15*t22*t23-l2*t11*t12*t13*t15*t17*t18-l2*t2*t11*t17*t18*t19*t23-l2*t2*t12*t17*t18*t19*t22+l2*t13*t15*t17*t18*t22*t23-l2*t2*t6*t11*t12*t15*t18*t19+l2*t2*t7*t11*t12*t15*t17*t19+l2*t2*t6*t15*t18*t19*t22*t23-l2*t2*t7*t15*t17*t19*t22*t23
    
    RA[0] = mt4
    RA[1] = mt5
    RA[2] = mt6
    RA[3] = 1.0
    
    
    b[0] = x + (l0*t19) / 2.0
    b[1] = y - (l0*t8*t13) / 2.0
    b[2] = z + (l0*t2*t8) / 2.0
    b[3] = 1.0

    rt[0] = t26-t28/2.0+t39/2.0+t49/2.0+t62/2.0-t63/2.0+x
    rt[1] = t30+t31+t37/2.0+t42/2.0-t43/2.0-t55/2.0-t65/2.0+t69/2.0+t76/2.0+t82/2.0-t83/2.0+y
    rt[2] = t32-t35/2.0+t46/2.0-t47/2.0+t52-t57/2.0-t59/2.0-t71/2.0-t73/2.0-t98/2.0+t99/2.0+z
    rt[3] = 1.0
    
    et1 = t30+t31+t37+t42+t69+t74+t76+t79+t82+t91+t102+y+(l2*t4*t8*t11*t12*t13)/2.0+(l2*t2*t6*t11*t18*t23)/2.0+(l2*t2*t6*t12*t18*t22)/2.0-(l2*t2*t7*t11*t17*t23)/2.0-(l2*t2*t7*t12*t17*t22)/2.0-(l2*t4*t8*t13*t22*t23)/2.0-(l2*t2*t6*t7*t11*t12*t15)/2.0+(l2*t2*t6*t7*t15*t22*t23)/2.0-(l2*t2*t11*t12*t15*t17*t18)/2.0+(l2*t6*t7*t11*t13*t19*t23)/2.0+(l2*t6*t7*t12*t13*t19*t22)/2.0+(l2*t2*t15*t17*t18*t22*t23)/2.0+(l2*t11*t13*t17*t18*t19*t23)/2.0+(l2*t12*t13*t17*t18*t19*t22)/2.0+(l2*t6*t11*t12*t13*t15*t18*t19)/2.0-(l2*t7*t11*t12*t13*t15*t17*t19)/2.0
    et2 = l2*t6*t13*t15*t18*t19*t22*t23*(-1.0/2.0)+(l2*t7*t13*t15*t17*t19*t22*t23)/2.0
    et3 = t32+t46+t52+t67+t77+t84+t86+t93+t94+t99+t103+z-(l2*t2*t4*t8*t11*t12)/2.0+(l2*t2*t4*t8*t22*t23)/2.0+(l2*t6*t11*t13*t18*t23)/2.0+(l2*t6*t12*t13*t18*t22)/2.0-(l2*t7*t11*t13*t17*t23)/2.0-(l2*t7*t12*t13*t17*t22)/2.0-(l2*t6*t7*t11*t12*t13*t15)/2.0-(l2*t2*t6*t7*t11*t19*t23)/2.0-(l2*t2*t6*t7*t12*t19*t22)/2.0+(l2*t6*t7*t13*t15*t22*t23)/2.0-(l2*t11*t12*t13*t15*t17*t18)/2.0-(l2*t2*t11*t17*t18*t19*t23)/2.0-(l2*t2*t12*t17*t18*t19*t22)/2.0+(l2*t13*t15*t17*t18*t22*t23)/2.0-(l2*t2*t6*t11*t12*t15*t18*t19)/2.0+(l2*t2*t7*t11*t12*t15*t17*t19)/2.0
    et4 = (l2*t2*t6*t15*t18*t19*t22*t23)/2.0-(l2*t2*t7*t15*t17*t19*t22*t23)/2.0;
    
    rc[0] = t26+t39+t49+t51+t62+t89+x-(l2*t4*t11*t12*t19)/2.0+(l2*t4*t19*t22*t23)/2.0+(l2*t6*t7*t8*t11*t23)/2.0+(l2*t6*t7*t8*t12*t22)/2.0+(l2*t8*t11*t17*t18*t23)/2.0+(l2*t8*t12*t17*t18*t22)/2.0+(l2*t6*t8*t11*t12*t15*t18)/2.0-(l2*t7*t8*t11*t12*t15*t17)/2.0-(l2*t6*t8*t15*t18*t22*t23)/2.0+(l2*t7*t8*t15*t17*t22*t23)/2.0
    rc[1] = et1+et2
    rc[2] = et3+et4
    rc[3] = 1.0
    
    lt[0] = t27*(-1.0/2.0)+t33+t38/2.0-t48/2.0-t60/2.0-t61/2.0+x
    lt[1] = t24+t36/2.0+t40/2.0+t41/2.0+t53+t54/2.0-t64/2.0+t68/2.0-t75/2.0-t80/2.0-t81/2.0+y
    lt[2] = t25+t29-t34/2.0+t44/2.0+t45/2.0+t56/2.0-t58/2.0-t70/2.0+t72/2.0+t96/2.0+t97/2.0+z
    lt[3] = 1.0
    
    et5 = t24+t36+t40+t41+t53+t54+t68+t90+t95+t100+t101+y+(l2*t3*t8*t9*t10*t13)/2.0+(l2*t2*t5*t9*t18*t21)/2.0+(l2*t2*t5*t10*t18*t20)/2.0+(l2*t2*t7*t9*t16*t21)/2.0+(l2*t2*t7*t10*t16*t20)/2.0-(l2*t3*t8*t13*t20*t21)/2.0+(l2*t2*t5*t7*t9*t10*t14)/2.0-(l2*t2*t5*t7*t14*t20*t21)/2.0-(l2*t2*t9*t10*t14*t16*t18)/2.0+(l2*t5*t7*t9*t13*t19*t21)/2.0+(l2*t5*t7*t10*t13*t19*t20)/2.0+(l2*t2*t14*t16*t18*t20*t21)/2.0-(l2*t9*t13*t16*t18*t19*t21)/2.0-(l2*t10*t13*t16*t18*t19*t20)/2.0-(l2*t5*t9*t10*t13*t14*t18*t19)/2.0-(l2*t7*t9*t10*t13*t14*t16*t19)/2.0
    et6 = (l2*t5*t13*t14*t18*t19*t20*t21)/2.0+(l2*t7*t13*t14*t16*t19*t20*t21)/2.0
    et7 = t25+t29+t44+t45+t56+t66+t72+t85+t92+t96+t97+z-(l2*t2*t3*t8*t9*t10)/2.0+(l2*t2*t3*t8*t20*t21)/2.0+(l2*t5*t9*t13*t18*t21)/2.0+(l2*t5*t10*t13*t18*t20)/2.0+(l2*t7*t9*t13*t16*t21)/2.0+(l2*t7*t10*t13*t16*t20)/2.0+(l2*t5*t7*t9*t10*t13*t14)/2.0-(l2*t2*t5*t7*t9*t19*t21)/2.0-(l2*t2*t5*t7*t10*t19*t20)/2.0-(l2*t5*t7*t13*t14*t20*t21)/2.0-(l2*t9*t10*t13*t14*t16*t18)/2.0+(l2*t2*t9*t16*t18*t19*t21)/2.0+(l2*t2*t10*t16*t18*t19*t20)/2.0+(l2*t13*t14*t16*t18*t20*t21)/2.0+(l2*t2*t5*t9*t10*t14*t18*t19)/2.0+(l2*t2*t7*t9*t10*t14*t16*t19)/2.0
    et8 = l2*t2*t5*t14*t18*t19*t20*t21*(-1.0/2.0)-(l2*t2*t7*t14*t16*t19*t20*t21)/2.0;
    lc[0] = t33+t38+t50+t78+t87+t88+x-(l2*t3*t9*t10*t19)/2.0+(l2*t3*t19*t20*t21)/2.0+(l2*t5*t7*t8*t9*t21)/2.0+(l2*t5*t7*t8*t10*t20)/2.0-(l2*t8*t9*t16*t18*t21)/2.0-(l2*t8*t10*t16*t18*t20)/2.0-(l2*t5*t8*t9*t10*t14*t18)/2.0-(l2*t7*t8*t9*t10*t14*t16)/2.0+(l2*t5*t8*t14*t18*t20*t21)/2.0+(l2*t7*t8*t14*t16*t20*t21)/2.0
    lc[1] = et5+et6
    lc[2] = et7+et8
    lc[3] = 1.0
    
    return B,H,LH,LK,LA,RH,RK,RA,b,rt,rc,lt,lc
