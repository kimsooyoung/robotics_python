import sympy as sy

m, g, l, I = sy.symbols('m g l I')
u1, u2 = sy.symbols('u1 u2')
x, y, phi = sy.symbols('x y phi')
x_d, y_d, phi_d = sy.symbols('x_d y_d phi_d')
x_dd, y_dd, phi_dd = sy.symbols('x_dd y_dd phi_dd')


T = (x_d**2 + y_d**2)*(m/2) + (phi_d**2)*(I/2)
V = m*g*y
L = T - V

H_g0 = sy.Matrix([
    [1, 0, x],
    [0, 1, y],
    [0, 0, 1]
])

H_01 = sy.Matrix([
    [sy.cos(phi), -sy.sin(phi), 0], 
    [sy.sin(phi),  sy.cos(phi), 0],
    [0, 0, 1]
])

H_g1 = H_g0*H_01

P = sy.Matrix([l/2, 0, 1])
R = sy.Matrix([-l/2, 0, 1])

P_0 = H_g1*P
R_0 = H_g1*R

P_0 = sy.Matrix([P_0[0], P_0[1]])
R_0 = sy.Matrix([R_0[0], R_0[1]])

R_op = sy.Matrix([
    [sy.cos(phi), -sy.sin(phi)],
    [sy.sin(phi), sy.cos(phi)]
])

F_op = R_op * sy.Matrix([0, u1])
F_or = R_op * sy.Matrix([0, u2])

q = sy.Matrix([x, y, phi])

J_p = P_0.jacobian(q)
J_r = R_0.jacobian(q)

Q = sy.simplify(J_p.T*F_op + J_r.T*F_or)

# EOM
q_d = sy.Matrix([x_d, y_d, phi_d])
d_dd = sy.Matrix([x_dd, y_dd, phi_dd])

EOM = []
dLdq = []
dLdqd = []
dt_dLdqd = []

for i in range(len(q)):
    dLdq.append(sy.diff(L, q[i]))
    dLdqd.append(sy.diff(L, q_d[i]))
    
    temp = 0
    for j in range(len(q)):
        temp += sy.diff(dLdqd[i], q[j])*q_d[j] + sy.diff(dLdqd[i], q_d[j])*d_dd[j]
    dt_dLdqd.append(temp)
    
    EOM.append(dt_dLdqd[i] - dLdq[i] - Q[i])
    
x_dd = sy.simplify(sy.solve(EOM[0],x_dd)[0])
y_dd = sy.simplify(sy.solve(EOM[1],y_dd)[0])
phi_dd = sy.simplify(sy.solve(EOM[2],phi_dd)[0])

print(f"x_dd = {x_dd}")
print(f"y_dd = {y_dd}")
print(f"phi_dd = {phi_dd}")