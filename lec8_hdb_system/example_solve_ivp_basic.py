import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def f(t, r):
    x, y = r

    fx = np.cos(y)
    fy = np.sin(x)

    return fx, fy


sol = integrate.solve_ivp(
    f, t_span=(0, 10), y0=(1, 1),
    t_eval=np.linspace(0, 10, 100)
)


x, y = sol.y
print(sol.y)

plt.plot(x, y)
plt.axis('scaled')
plt.show()
