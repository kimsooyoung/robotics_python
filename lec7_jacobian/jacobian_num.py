import numpy as np

def func(x, y):
    return np.array([[x**2+y**2], [2*x+3*y+5]])

z = np.array([1, 2])
f = func(z[0], z[1])
epsilon = 1e-3

# J = ([
#     [df1/dx, df1/dy],
#     [df2/dx, df2/dy]
# ])
J = np.eye(2)

# x
dfdx = (func(z[0] + epsilon, z[1]) - func(z[0], z[1])) / epsilon
# dfdx.shape => 2 * 1 [[],[]]
J[0, 0] = dfdx[0, 0]
J[1, 0] = dfdx[1, 0]

dfdy = (func(z[0], z[1] + epsilon) - func(z[0], z[1])) / epsilon
J[0, 1] = dfdy[0, 0]
J[1, 1] = dfdy[1, 0]

print(J)