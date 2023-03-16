import sympy as sy

t0, tf  = sy.symbols('t0 tf', real=True) # defining the variables
q0, qf  = sy.symbols('q0 qf', real=True) # defining the variables

A = sy.Matrix([ [ 1, t0],
                  [1, tf]
                 ])
b = sy.Matrix([ [q0],
                [qf]
                 ])

Ainv = A.inv()
x = Ainv*b
print(x)
# print(sy.solve(EOM[0],xddot))
# x = simplify(inv(A)*b)
