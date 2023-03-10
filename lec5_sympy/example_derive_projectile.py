import sympy as sy

x,y = sy.symbols("x y", real=True)
x_d, y_d = sy.symbols("x_d y_d", real=True)
x_dd, y_dd = sy.symbols("x_dd y_dd", real=True)
m,c,g  = sy.symbols('m c g', real=True)

v = sy.sqrt(x_d ** 2 + y_d ** 2)
Fx = -c * x_d * v
Fy = -c * y_d * v

T = 0.5 * m * (x_d ** 2 + y_d ** 2)
V = m * g * y
L = T - V

dL_dx_d = sy.diff(L, x_d)
dt_dL_dx_d = sy.diff(dL_dx_d, x) * x_d + \
             sy.diff(dL_dx_d, x_d) * x_dd + \
             sy.diff(dL_dx_d, y) * y_d + \
             sy.diff(dL_dx_d, y_d) * y_dd
dL_dx   = sy.diff(L, x)
EOM1 = dt_dL_dx_d - dL_dx - Fx
print(EOM1)

dL_dy_d = sy.diff(L, y_d)
dt_dL_dy_d = sy.diff(dL_dy_d, x) * x_d + \
             sy.diff(dL_dy_d, x_d) * x_dd + \
             sy.diff(dL_dy_d, y) * y_d + \
             sy.diff(dL_dy_d, y_d) * y_dd
dL_dy = sy.diff(L, y)
EOM2 = dt_dL_dy_d - dL_dy - Fy
print(EOM2)

print(sy.solve(EOM1, x_dd))
print(sy.solve(EOM2, y_dd))
print('\n')

q = [x, y]
q_d = [x_d, y_d]
q_dd = [x_dd, y_dd]
F = [Fx, Fy]

dL_dq_d = []
dt_dL_dq_d = []
dL_dq = []
EOM = []

for i in range(len(q)):
    dL_dq_d.append(sy.diff(L, q_d[i]))
    temp_symbol = 0
    for j in range(len(q)):
        temp_symbol += sy.diff(dL_dq_d[i], q[j]) * q_d[j] + \
                       sy.diff(dL_dq_d[i], q_d[j]) * q_dd[j]
    
    dt_dL_dq_d.append(temp_symbol)
    dL_dq.append(sy.diff(L, q[i]))
    EOM.append(dt_dL_dq_d[i] - dL_dq[i] - F[i])

print(sy.solve(EOM[0], x_dd))
print(sy.solve(EOM[1], y_dd))