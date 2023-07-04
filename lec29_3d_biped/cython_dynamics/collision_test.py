import numpy as np
import collision
import collision_cython

z_in = np.random.rand(28)
params = np.random.rand(3)
t = 0.1

g_stop = collision_cython.collision(t, z_in, params)

print(f"g_stop: {g_stop}\n")

def cython_main():

    # collision_cython.collision.terminal = True
    # collision_cython.collision.direction = 0
        
    z_in = np.random.rand(28)
    params = np.random.rand(3)
    t = 0.1
    
    g_stop = collision_cython.collision(t, z_in, params)    

def python_main():
    
    z_in = np.random.rand(28)
    params = np.random.rand(3)
    t = 0.1
    
    g_stop = collision.collision(t, z_in, params)    

import timeit

print(f"cython: {timeit.timeit(cython_main, number=10)}")
print(f"python: {timeit.timeit(python_main, number=10)}")

### Result ###
# cython: 0.0011163100000000092
# python: 11.327367436

