import sympy as sy

t  = sy.symbols('t', real=True) # defining the variables
a10, a11, a12, a13  = sy.symbols('a10 a11 a12 a13', real=True) # defining the variables
a20, a21, a22, a23  = sy.symbols('a20 a21 a22 a23', real=True) # defining the variables

pi = sy.pi

#segment 1
#initial time, final time
t1_0, t1_N = 0, 1.5
#initial position, final position
theta1_0, theta1_N = 0, 0.5*pi 

# segment 2
t2_0, t2_N = 1.5, 3
theta2_0, theta2_N = theta1_N, 0;

theta1 = a10+a11*t+a12*t**2+a13*t**3;
theta2 = a20+a21*t+a22*t**2+a23*t**3;
theta1dot, theta2dot = sy.diff(theta1,t), sy.diff(theta2,t)
theta1ddot, theta2ddot = sy.diff(theta1dot,t), sy.diff(theta2dot,t)

# %various eqns
# theta1(0) = 0 / theta1(1.5) = 0.5*pi
# theta2(1.5) = 0.5*pi / theta(3) = 0
# theta1_d(0) = 0 / theta2_d(3) = 0
# theta1_d(1.5) = theta2_d(1.5) / theta1_dd(1.5) = theta2_dd(1.5)
eqn1 = theta1.subs(t,0) - theta1_0;
eqn2 = theta1.subs(t,1.5) - theta1_N;
eqn3 = theta2.subs(t,1.5) - theta2_0;
eqn4 = theta2.subs(t,3) - theta2_N;
eqn5 = theta1dot.subs(t,0) - 0;
eqn6 = theta2dot.subs(t,3) - 0;
eqn7 = theta1dot.subs(t,1.5)-theta2dot.subs(t,1.5);
eqn8 = theta1ddot.subs(t,1.5)-theta2ddot.subs(t,1.5);

# %%We want to write the above equations at A x = b
# %where x = [a10 a11 a12 a13 a20 a21 a22 a23 ];
# %A = coefficients of a10 a11 a12 a13 a20 a21 a22 a23; b=constants coefficients;
q = sy.Matrix([a10, a11, a12, a13, a20, a21, a22, a23])
eqn = sy.Matrix([eqn1,eqn2,eqn3,eqn4,eqn5,eqn6,eqn7,eqn8])
# 결국 A는 계수만 뽑아낸 행렬이다. 이전에 했던 거랑 동일
A = eqn.jacobian(q)
# b = Matrix([[0], [0.5*pi], [0.5*pi], [0], [0], [0], [0], [0]])
b = -eqn.subs([ (a10,0), (a11,0),(a12,0), (a13,0), \
                (a20,0), (a21,0),(a22,0), (a23,0)])

#print(A)
Ainv = A.inv()
x = Ainv*b

print(f't1_0 = {t1_0} / t1_N = {t1_N}')
print(f't2_0 = {t1_0} / t2s_N = {t1_N} \n')
print('a10 = ',x[0])
print('a11 = ',x[1])
print('a12 = ',x[2])
print('a13 = ',x[3])
print('a20 = ',x[4])
print('a21 = ',x[5])
print('a22 = ',x[6])
print('a23 = ',x[7])
