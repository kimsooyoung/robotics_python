import numpy as np

A = np.array([
    [1,2,3],
    [3,4,5],
    [6,7,8]
])

b = np.array([
    1,2,3
])

print(A, b, b.T)
print(np.linalg.solve(A, b.T))