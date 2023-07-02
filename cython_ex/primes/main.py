import timeit

import primes_cython
print(timeit.timeit("primes_cython.primes(10)", setup="import primes_cython", number=100))

import primes_python_compiled
print(timeit.timeit("primes_python_compiled.primes(10)", setup="import primes_python_compiled", number=100))