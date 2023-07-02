import numpy as np
import convolve_py
import convolve1_wo_optim
import convolve1
import timeit

# Usage example:
# convolve_py.naive_convolve(
#     np.array([[1, 1, 1]], dtype=np.int64),
#     np.array([[1],[2],[1]], dtype=np.int64)
# )

# convolve1.naive_convolve(
#     np.array([[1, 1, 1]], dtype=np.int64),
#     np.array([[1],[2],[1]], dtype=np.int64)
# )

N = 100
f = np.arange(N*N, dtype=np.int64).reshape((N,N))
g = np.arange(81, dtype=np.int64).reshape((9, 9))

print(f"convolve_py : {timeit.timeit('convolve_py.naive_convolve(f, g)', setup='import numpy as np; from __main__ import convolve_py, f, g', number=10)}")
print(f"convolve1_wo_optim : {timeit.timeit('convolve1_wo_optim.naive_convolve(f, g)', setup='import numpy as np; from __main__ import convolve1_wo_optim, f, g', number=10)}")
print(f"convolve1 : {timeit.timeit('convolve1.naive_convolve(f, g)', setup='import numpy as np; from __main__ import convolve1, f, g', number=10)}")

import compute_memview
array_1 = np.random.uniform(0, 1000, size=(3000, 2000)).astype(np.intc)
array_2 = np.random.uniform(0, 1000, size=(3000, 2000)).astype(np.intc)
a = 4
b = 3
c = 9

compute_memview.compute(array_1, array_2, a, b, c)