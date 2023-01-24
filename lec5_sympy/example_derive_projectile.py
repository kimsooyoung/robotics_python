import sympy as sy

x,y  = sy.symbols('x y', real=True)
xdot,ydot  = sy.symbols('xdot ydot', real=True)
xddot,yddot  = sy.symbols('xddot yddot', real=True)
m,c,g  = sy.symbols('m c g', real=True)

#1) position and velocity
#positions are x and y
#velocities are xdot and yddot

#2) Kinetic and potential energy
T = 0.5*m*(xdot*xdot + ydot*ydot)
V = m*g*y
L = T-V
# print(type(T))
# print(type(V))
# print(type(L))

#3) Euler-Lagrange equations
v = sy.sqrt(xdot*xdot + ydot*ydot)
Fx = -c*xdot*v
Fy = -c*ydot*v

#4a) EOM: Manually writing down all terms
# Since L(x,y,xdot,ydot)
dLdxdot = sy.diff(L,xdot)
ddt_dLdxdot = sy.diff(dLdxdot,x)*xdot + sy.diff(dLdxdot,xdot)*xddot + \
              sy.diff(dLdxdot,y)*ydot + sy.diff(dLdxdot,ydot)*yddot
dLdx = sy.diff(L,x)
EOM1 = ddt_dLdxdot - dLdx -Fx
# print(EOM1)

dLdydot = sy.diff(L,ydot)
ddt_dLdydot = sy.diff(dLdydot,x)*xdot + sy.diff(dLdydot,xdot)*xddot + \
              sy.diff(dLdydot,y)*ydot + sy.diff(dLdydot,ydot)*yddot
dLdy = sy.diff(L,y)
EOM2 = ddt_dLdydot - dLdy -Fy

print(sy.solve(EOM1,xddot))
print(sy.solve(EOM2,yddot))
print('\n')

#4a) EOM: Using loops to automate equation generation
q = [x, y]
qdot = [xdot, ydot]
qddot = [xddot, yddot]
F = [Fx, Fy]
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
