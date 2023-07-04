import numpy as np
import humanoid_rhs_cython

z_in = np.random.rand(28)
params = np.random.rand(17)
t = 0.1


A_ss, b_ss, J_l, J_r, Jdot_l, Jdot_r = humanoid_rhs_cython.humanoid_rhs(z_in, t, params)

print(f"A_ss: {A_ss}")
print(f"b_ss: {b_ss}")
print(f"J_l: {J_l}")
print(f"J_r: {J_r}")
print(f"Jdot_l: {Jdot_l}")
print(f"Jdot_r: {Jdot_r}\n")

print(f"size(A_ss) : {A_ss.shape} ")
print(f"size(b_ss) : {b_ss.shape} ")
print(f"size(J_l) : {J_l.shape} ")
print(f"size(J_r) : {J_r.shape} ")
print(f"size(Jdot_l) : {Jdot_l.shape} ")
print(f"size(Jdot_r) : {Jdot_r.shape} ")