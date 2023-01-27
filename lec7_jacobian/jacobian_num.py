import numpy as np

def fun(z):
    print(z)
    x = z[0]
    y = z[1]
    f = np.array([[x**2+y**2], [2*x+3*y+5]])
    return f

z0 = np.array([1, 2])
pert = 1e-3

J = np.zeros((2,2))
ztemp1 = np.array([z0[0]+pert, z0[1]])
Jtemp_column1 = (fun(ztemp1)-fun(z0))/pert #first column
J[0,0] = Jtemp_column1[0]
J[1,0] = Jtemp_column1[1]

ztemp2 = np.array([z0[0], z0[1]+pert])
Jtemp_column2 = (fun(ztemp2)-fun(z0))/pert #second column
J[0,1] = Jtemp_column2[0]
J[1,1] = Jtemp_column2[1]

print(J)
