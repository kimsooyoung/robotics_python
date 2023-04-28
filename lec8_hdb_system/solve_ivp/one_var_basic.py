from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

# differential equation
def dydt(t, y):
    return -y + 2*np.sin(t)

ts = np.linspace(0, 10, 100)
initial_state = (1, 1)
sol = solve_ivp(dydt, t_span=(0, 10), y0=[10], t_eval=ts)

t = sol.t
y = sol.y[0]

plt.plot(t, y)
plt.show()