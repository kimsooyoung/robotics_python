import sympy as sy

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


q_d = sy.Matrix([phidot,thetadot,psidot])


om = psidot*k + R_z*(thetadot*j)+ R_z*R_y*(phidot*i)
# 하나의 기법으로 표현
# 결국 구하는 것은 R_w, R_wb 행렬이다.
# w = R_we * q_dot임을 시용해서 R_we만 뽑아내는 것이다.
# R_we = om.jacobian([phidot,thetadot,psidot])
R_we = om.jacobian(q_d)

print(sy.simplify(R_we))
print(sy.simplify(R_we.det()))

om_b = phidot*i +  R_x.transpose()*(thetadot*j) + R_x.transpose()*R_y.transpose()*(psidot*k);
R_be = om_b.jacobian(q_d)

print(sy.simplify(R_be))
print(sy.simplify(R_be.det()))