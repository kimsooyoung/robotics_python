import sympy as sy

t  = sy.symbols('t', real=True) # defining the variables
a10, a11, a12, a13  = sy.symbols('a10 a11 a12 a13', real=True) # defining the variables

pi = sy.pi

#segment 1
t1_0 = 0; #initial time
t1_N = 3.0; #final time
theta1_0 = -0.5*pi-0.5; #initial position
theta1_N = -0.5*pi+0.5; #final position

theta1 = a10+a11*t+a12*t**2+a13*t**3;
theta1dot = sy.diff(theta1,t);
theta1ddot = sy.diff(theta1dot,t);

# %various eqns
# 시간 나누지 않아서 equ수가 작다.
# theta(0) = -0.5*pi-0.5 / theta(3) = -0.5*pi+0.5
# omega(0) = 0 / omega(3) = 0
eqn1 = theta1.subs(t,t1_0) -theta1_0;
eqn2 = theta1.subs(t,t1_N) -theta1_N;
eqn3 = theta1dot.subs(t,t1_0)-0;
eqn4 = theta1dot.subs(t,t1_N)-0;

# %%We want to write the above equations at A x = b
# %where x = [a10 a11 a12 a13 a20 a21 a22 a23 ];
# %A = coefficients of a10 a11 a12 a13 a20 a21 a22 a23; b=constants coefficients;
q = sy.Matrix([a10, a11, a12, a13])
eqn = sy.Matrix([eqn1,eqn2,eqn3,eqn4])
A = eqn.jacobian(q)

b = -eqn.subs([ (a10,0), (a11,0),(a12,0), (a13,0)])

#print(A)
Ainv = A.inv()
x = Ainv*b
# print(x)

print('t1_0 = ',t1_0)
print('t1_N = ',t1_N)
print('\n')
print('a10 = ',x[0])
print('a11 = ',x[1])
print('a12 = ',x[2])
print('a13 = ',x[3])
