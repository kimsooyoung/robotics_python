import sympy as sy
import matrix_helper as mh

theta1, theta2 = sy.symbols("theta1 theta2", real=True)
omega1, omega2 = sy.symbols("omega1 omega2", real=True)
alpha1, alpha2 = sy.symbols("alpha1 alpha2", real=True)

link1, link2 = sy.symbols("link1 link2", real=True)
c1, c2 = sy.symbols("c1 c2", real=True)
I1, I2 = sy.symbols("I1 I2", real=True)
m1, m2 = sy.symbols("m1 m2", real=True)
g = sy.symbols("g", real=True)

# link1 frame to world frame
# caution! 
# 1.5 * sy.pi => auto caculation not works
H_01 = mh.calc_homogeneous_2d(0, 0, 3 * sy.pi / 2 + theta1)
H_12 = mh.calc_homogeneous_2d(link1, 0, theta2)

# c1 in world frame 
G1 = H_01 * sy.Matrix([c1, 0, 1])
G2 = H_01 * H_12 * sy.Matrix([c2, 0, 1])

G1_x = sy.Matrix([G1[0]])
G1_y = sy.Matrix([G1[1]])
G2_x = sy.Matrix([G2[0]])
G2_y = sy.Matrix([G2[1]])

# print(G1_x)
# print(G1_y)
# print(G2_x)
# print(G2_y)

# velocity vectors
q = sy.Matrix([theta1, theta2])
q_d = sy.Matrix([omega1, omega2])
v_G1_x = G1_x.jacobian(q) * q_d
v_G1_y = G1_y.jacobian(q) * q_d
v_G2_x = G2_x.jacobian(q) * q_d
v_G2_y = G2_y.jacobian(q) * q_d

v_G1 = sy.Matrix([v_G1_x, v_G1_y])
v_G2 = sy.Matrix([v_G2_x, v_G2_y])

#2) Lagrangian
T = 0.5 * m1 * v_G1.dot(v_G1) + \
    0.5 * m2 * v_G2.dot(v_G2) + \
    0.5 * I1 * omega1 * omega1 + \
    0.5 * I2 * (omega1 + omega2) * (omega1 + omega2)
V = m1 * g * G1[1] + m2 * g * G2[1]

L = T - V

print(L)
print(f"T: {T}")
print(f"V: {V}")

#3) Derive EOM
# q, q_d는 앞서 velocity에서 이미 해줬음
q_dd = sy.Matrix([alpha1, alpha2]) #thetaddot

dL_dq_d = []
dt_dL_dq_d = []
dL_dq = []
EOM = []

for i in range(len(q)):
    dL_dq_d.append(sy.diff(L, q_d[i]))
    temp = 0
    for j in range(len(q)):
        temp += sy.diff(dL_dq_d[i], q[j]) * q_d[j] + \
                sy.diff(dL_dq_d[i], q_d[j]) * q_dd[j]
    
    dt_dL_dq_d.append(temp)
    dL_dq.append(sy.diff(L, q[i]))
    # 현재 외력이 0이므로 이 두개 항만 있다.
    EOM.append(dt_dL_dq_d[i] - dL_dq[i])


print(sy.solve(EOM[0], alpha1))
print(sy.solve(EOM[1], alpha2))

# C : 코리올리
# G : gravity?
# M(q)*q_dd + C(q, q_d)*q_d + G(q) -Tau = 0
# b = C(q, q_d)*q_d + G(q) -Tau
# G = G(q)
# C = b - G = C(q, q_d)*q_d + G(q) - G(q) = C(q, q_d)*q_d
EOM = sy.Matrix([EOM[0],EOM[1]])
# EOM에서 alpha 들어간 term만 뽑아내는 방법론임
# alpha1, alpha2로 자코비안 계산하면, 2 * 2인 M 행렬을 바로 뽑을 수 있다. 
M = EOM.jacobian(q_dd)
print(f"EOM: {EOM}")
print(f"M: {M}")

b1 = EOM[0].subs([ (alpha1,0), (alpha2,0)])
b2 = EOM[1].subs([ (alpha1,0), (alpha2,0)])
G1 = b1.subs([ (omega1,0), (omega2,0)])
G2 = b2.subs([ (omega1,0), (omega2,0)])
C1 = b1 - G1
C2 = b2 - G2

print('M11 = ', sy.simplify(M[0,0]))
print('M12 = ', sy.simplify(M[0,1]))
print('M21 = ', sy.simplify(M[1,0]))
print('M22 = ', sy.simplify(M[1,1]),'\n')

print('C1 = ', sy.simplify(C1))
print('C2 = ', sy.simplify(C2),'\n')
print('G1 = ', sy.simplify(G1))
print('G2 = ', sy.simplify(G2),'\n')
