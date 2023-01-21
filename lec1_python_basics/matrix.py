import numpy as np

A = np.array([[2,4],[5,-6]])
print(A)

# rotation matrix generation func
def rot_mat(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],    
    ]) 

theta = 0.5
print(rot_mat(theta))

# (2 X 2) dot (2 X 1) => (2 X 1)
B = np.array([[2],[2]])
D = A.dot(B)
D_ = A @ B
print(D)
print(B)

# Transpose
print(A.transpose())
# or
print(np.transpose(A))

# inverse
inv_a = np.linalg.inv(A)
print(inv_a)

# Element-wise mult & Matrix mult
print(inv_a * A)
print()
print(np.matmul(inv_a, A))
print(inv_a.dot(A))
print(inv_a @ A)

# identity matrix
print(np.identity(5))