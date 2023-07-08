import numpy as np
import humanoid_rhs_cython
import humanoid_rhs

z_in = np.random.rand(28)
params = np.random.rand(17)
t = 0.1


A_ss, b_ss, J_l, J_r, Jdot_l, Jdot_r = humanoid_rhs_cython.humanoid_rhs(z_in, t, params)

# print(f"A_ss: {A_ss}")
# print(f"b_ss: {b_ss}")
# print(f"J_l: {J_l}")
# print(f"J_r: {J_r}")
# print(f"Jdot_l: {Jdot_l}")
# print(f"Jdot_r: {Jdot_r}\n")

# print(f"size(A_ss) : {A_ss.shape} ")
# print(f"size(b_ss) : {b_ss.shape} ")
# print(f"size(J_l) : {J_l.shape} ")
# print(f"size(J_r) : {J_r.shape} ")
# print(f"size(Jdot_l) : {Jdot_l.shape} ")
# print(f"size(Jdot_r) : {Jdot_r.shape} \n")

def cython_main():
    
    z_in = np.random.rand(28)
    params = np.random.rand(17)
    t = 0.1
    
    A_ss, b_ss, J_l, J_r, Jdot_l, Jdot_r = humanoid_rhs_cython.humanoid_rhs(z_in, t, params)
    
def python_main():
    
    z_in = np.random.rand(28)
    params = np.random.rand(17)
    t = 0.1
    
    A_ss, b_ss, J_l, J_r, Jdot_l, Jdot_r = humanoid_rhs.humanoid_rhs(z_in, t, params)

import timeit

print(f"cython: {timeit.timeit(cython_main, number=10)}")
print(f"python: {timeit.timeit(python_main, number=10)}")

### Result ###
# cython: 0.0011163100000000092
# python: 11.327367436

### Debug ### 
t = 0.0
z_in = np.array([
    -0.000049815000110,
    1.014525925343273,
    0.070715370139685,
    -0.001725772505528,
    0.999571112013753,
    0.000000000000035,
                    0,
                    0,
                    0,
    0.000000000000001,
    -0.000000000000006,
    -0.000000000000003,
    0.000000000000006,
    0.000000000000043,
    -0.000000000000002,
    0.000000000000070,
    -0.000000000000013,
    -0.000000000000057,
    -0.999999999999981,
    0.000000000000061,
    -0.029247773468329,
    0.054577886084082,
    -0.001551040824746,
    -1.029460778006675,
    0.054690088280553,
    0.559306810759302,
    -0.000000000000001,
    -0.000000000000031    
])
params = np.array([
    70.,   
    10.,    
    5.,    
    5.,    
    3.,    
    2.,    
    1.,    
    0.3,   
    2.,    
    0.5,   
    0.15,  
    1.,
    1.,    
    0.5,   
    0.5,   
    0.1,   
    9.8 
])

A_ss, b_ss, J_l, J_r, Jdot_l, Jdot_r = humanoid_rhs_cython.humanoid_rhs(z_in, t, params)

print(f"A_ss : {A_ss}")
print(f"b_ss : {b_ss}")
print(f"J_l : {J_l}")
print(f"J_r : {J_r}")
print(f"Jdot_l : {Jdot_l}")
print(f"Jdot_r : {Jdot_r}")
