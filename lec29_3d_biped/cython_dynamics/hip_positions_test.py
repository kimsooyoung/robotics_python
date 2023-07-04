import numpy as np
import hip_positions
import hip_positions_cython

z_in = np.random.rand(14)

hip_position = hip_positions_cython.hip_positions(
    z_in[0], z_in[1], z_in[2], z_in[3], z_in[4], z_in[5],
    z_in[6], z_in[7], z_in[8], z_in[9], z_in[10], z_in[11],
    z_in[12], z_in[13]
)

print(f"hip_position: {hip_position}\n")

def cython_main():

    z_in = np.random.rand(14)

    hip_position = hip_positions_cython.hip_positions(
        z_in[0], z_in[1], z_in[2], z_in[3], z_in[4], z_in[5],
        z_in[6], z_in[7], z_in[8], z_in[9], z_in[10], z_in[11],
        z_in[12], z_in[13]
    )

def python_main():
    
    z_in = np.random.rand(14)

    hip_position = hip_positions.hip_positions(
        z_in[0], z_in[1], z_in[2], z_in[3], z_in[4], z_in[5],
        z_in[6], z_in[7], z_in[8], z_in[9], z_in[10], z_in[11],
        z_in[12], z_in[13]
    )

import timeit

print(f"cython: {timeit.timeit(cython_main, number=10)}")
print(f"python: {timeit.timeit(python_main, number=10)}")

### Result ###
# cython: 0.0011163100000000092
# python: 11.327367436

