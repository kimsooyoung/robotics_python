import sympy as sy

t0, tf  = sy.symbols('t0 tf', real=True) # defining the variables
q0, qf  = sy.symbols('q0 qf', real=True) # defining the variables

A = sy.Matrix([ [1, t0, t0**2, t0**3],
                [1, tf, tf**2, tf**3],
                [0,  1, 2*t0,  3*t0**2],
                [0,  1, 2*tf,  3*tf**2],
                 ])
b = sy.Matrix([ [q0],
                [qf],
                [0],
                [0]
                 ])


Ainv = A.inv()
x = Ainv*b
print(x)
