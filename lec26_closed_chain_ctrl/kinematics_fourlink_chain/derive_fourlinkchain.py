import sympy as sy

q1, q2, q3, q4 = sy.symbols('q1 q2 q3 q4')
u1, u2, u3, u4 = sy.symbols('u1 u2 u3 u4')
a1, a2, a3, a4 = sy.symbols('a1 a2 a3 a4')
m1, m2, m3, m4 = sy.symbols('m1 m2 m3 m4')
I1, I2, I3, I4 = sy.symbols('I1 I2 I3 I4')
l1, l2, l3, l4 = sy.symbols('l1 l2 l3 l4')
T1, T2, T3, T4 = sy.symbols('T1 T2 T3 T4')
lx, ly = sy.symbols('lx ly')
g = sy.symbols('g')

### position vectors ###
def Homogeneous(angle, x, y):
    return sy.Matrix([
        [sy.cos(angle), -sy.sin(angle), x],
        [sy.sin(angle), sy.cos(angle),  y],
        [0,   0,  1]
    ])

H01 = Homogeneous( 3*sy.pi/2 + q1, 0, 0)
H12 = Homogeneous( q2, l1, 0)
H02 = H01*H12

H03 = Homogeneous( 3*sy.pi/2 + q3, lx, ly)
H34 = Homogeneous( q4, l3, 0)
H04 = H03*H34

G1 = H01 * sy.Matrix([0.5*l1, 0, 1])
G2 = H02 * sy.Matrix([0.5*l2, 0, 1])
G3 = H03 * sy.Matrix([0.5*l3, 0, 1])
G4 = H04 * sy.Matrix([0.5*l4, 0, 1])

G1_xy = sy.Matrix([G1[0], G1[1]])
G2_xy = sy.Matrix([G2[0], G2[1]])
G3_xy = sy.Matrix([G3[0], G3[1]])
G4_xy = sy.Matrix([G4[0], G4[1]])

### end of last link is P2 and P4 (say) then position and velocity is given by ###
P2 = H02 * sy.Matrix([l2, 0, 1])
P4 = H04 * sy.Matrix([l4, 0, 1])
del_x = sy.simplify(P2[0] - P4[0])
del_y = sy.simplify(P2[1] - P4[1])

leg_length = sy.sqrt(P2[0]**2 + P2[1]**2)
leg_angle = 0.5 * (q1+q3)

print(f"del_x = {del_x}")
print(f"del_y = {del_y}")
print(f"leg_length = {sy.simplify(leg_length)}")
print(f"leg_angle = {leg_angle}")
