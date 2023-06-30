import sympy as sy
import numpy as np

x, y, z = sy.symbols('x y z')

a = sy.Matrix([
    [x],
    [y],
    [z],
    [0]
])

print(a.dot(a))
print(a.shape)

ll = [1,2,3,4]

lm = sy.Matrix(ll)

print(lm)

X_des = np.zeros((8,1))

for i in range(8):
    X_des[i] = i
    
print(X_des)

k = np.array([1,2,3])
k = np.hstack(( np.zeros(6) , k))
print(k)