# example of single threaded matrix solver
from os import environ

from time import time
from numpy.random import rand
from numpy.linalg import solve

# record the start time
start = time()

# size of arrays
n = 8000
# create matrix
a = rand(n, n)
# create result
b = rand(n, 1)

# solve least squares equation
x = solve(a, b)
# calculate and report duration
duration = time() - start

print(f'Took {duration:.3f} seconds')
# 3.182