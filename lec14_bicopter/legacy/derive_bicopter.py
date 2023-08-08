import sympy as sy

x,y,phi  = sy.symbols('x y phi', real=True)
xdot,ydot,phidot  = sy.symbols('xdot ydot phidot', real=True)
xddot,yddot,phiddot  = sy.symbols('xddot yddot phiddot', real=True)
m,I,g,l  = sy.symbols('m I g l', real=True)
u1, u2 = sy.symbols('u1 u2', real=True)
#1) position and velocity
#positions are x and y
#velocities are xdot and yddot

#2) Kinetic and potential energy
T = 0.5*m*(xdot*xdot + ydot*ydot) + 0.5*I*phidot*phidot
V = m*g*y
L = T-V
# print(type(T))
# print(type(V))
# print(type(L))

f1 = sy.Matrix([x+0.5*l*sy.cos(phi),y+0.5*l*sy.sin(phi)])
z1 = sy.Matrix([x, y, phi])
J1 = f1.jacobian(z1)
#print(J1)

f2 = sy.Matrix([x-0.5*l*sy.cos(phi),y-0.5*l*sy.sin(phi)])
z2 = sy.Matrix([x, y, phi])
J2 = f2.jacobian(z2)
#print(J2)

F1 = sy.Matrix([-u1*sy.sin(phi),u1*sy.cos(phi)])
F2 = sy.Matrix([-u2*sy.sin(phi),u2*sy.cos(phi)])

F = sy.simplify(sy.Transpose(J1)*F1+sy.Transpose(J2)*F2)
print(F)

# #3) Euler-Lagrange equations
Fx = F[0]
Fy = F[1]
tau = F[2]


#4 EOM: Using loops to automate equation generation
q = [x, y, phi]
qdot = [xdot, ydot, phidot]
qddot = [xddot, yddot, phiddot]
F = [Fx, Fy, tau]
dLdqdot = []
ddt_dLdqdot = []
dLdq = []
EOM = []
mm = len(qddot)
for i in range(0,mm):
    dLdqdot.append(sy.diff(L,qdot[i]))
    tmp = 0;
    for j in range(0,mm):
        tmp += sy.diff(dLdqdot[i],q[j])*qdot[j]+ sy.diff(dLdqdot[i],qdot[j])*qddot[j]
    ddt_dLdqdot.append(tmp)
    dLdq.append(sy.diff(L,q[i]));
    EOM.append(ddt_dLdqdot[i] - dLdq[i] - F[i])

print(sy.solve(EOM[0],xddot))
print(sy.solve(EOM[1],yddot))
print(sy.solve(EOM[2],phiddot))
