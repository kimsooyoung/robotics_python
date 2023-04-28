from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

def f(t, r):
    x, y = r
    
    fx = np.cos(y)
    fy = np.sin(x)
    
    return fx, fy

sol = integrate.solve_ivp(
    f, t_span=(0,10), y0=(1,1),
    t_eval=np.linspace(0,10,100),
)

t = sol.t
x, y = sol.y
plt.plot(x, y)
plt.axis("scaled")
plt.show()
