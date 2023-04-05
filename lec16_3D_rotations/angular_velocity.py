import sympy as sy
import numpy as np

def sin(angle):
    return sy.sin(angle)

def cos(angle):
    return sy.cos(angle)


phi, theta, psi = sy.symbols('phi theta psi', real=True)
phidot, thetadot, psidot  = sy.symbols('phidot thetadot psidot', real=True)

# %%%%%%% unit vectors %%%%%%%
i = sy.Matrix([1, 0, 0]);
j = sy.Matrix([0, 1, 0]);
k = sy.Matrix([0, 0, 1]);

# %%%%%% rotation vectors %%%%%
R_x = sy.Matrix([ [1, 0, 0],
                  [0, cos(phi),  -sin(phi)],
                  [0, sin(phi),   cos(phi)] ])

R_y = sy.Matrix( [ [cos(theta),  0,   sin(theta)],
                   [0,          1,         0    ],
                   [-sin(theta), 0,   cos(theta)]])

R_z = sy.Matrix( [ [cos(psi), -sin(psi),   0],
                   [sin(psi),  cos(psi),   0],
                   [0,           0,         1] ])

om = psidot*k + R_z*(thetadot*j)+ R_z*R_y*(phidot*i)

R_we = om.jacobian([phidot,thetadot,psidot])
print(sy.simplify(R_we.det()))

om_b = phidot*i +  R_x.transpose()*(thetadot*j) + R_x.transpose()*R_y.transpose()*(psidot*k);
R_be = om_b.jacobian([phidot,thetadot,psidot]);

print(sy.simplify(R_be.det()))
