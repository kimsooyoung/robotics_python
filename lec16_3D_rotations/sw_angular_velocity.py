import sympy as sy

def sin(angle):
    return sy.sin(angle)

def cos(angle):
    return sy.cos(angle)

phi, theta, psi = sy.symbols("phi theta psi", real=True)
phi_d, theta_d, psi_d = sy.symbols("phi_d theta_d psi_d", real=True)

R_x = sy.Matrix([
    [1,        0,         0],
    [0, cos(phi), -sin(phi)],
    [0, sin(phi),  cos(phi)]
])

R_y = sy.Matrix([
    [cos(theta),  0, sin(theta)],
    [0,           1,          0],
    [-sin(theta), 0, cos(theta)]
])

R_z = sy.Matrix([
    [cos(psi), -sin(psi), 0],
    [sin(psi),  cos(psi), 0],
    [0,                0, 1]
])

i = sy.Matrix([1, 0, 0])
j = sy.Matrix([0, 1, 0])
k = sy.Matrix([0, 0, 1])

q_d = sy.Matrix([phi_d, theta_d, psi_d])

w = psi_d*k + theta_d*(R_z*j) + phi_d*(R_z*R_y*i)
R_w = w.jacobian(q_d)
print("R_w")
print(sy.simplify(R_w))

w_b = phi_d*i + theta_d*(R_x.T*j) + psi_d*(R_x.T*R_y.T*k)
R_wb = w_b.jacobian(q_d)
print("R_wb")
print(sy.simplify(R_wb))

print("R_w.det: ", end='')
print(sy.simplify(R_w.det()))
print("R_wb.det: ", end='')
print(sy.simplify(R_wb.det()))