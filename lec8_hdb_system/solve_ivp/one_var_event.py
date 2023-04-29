from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

# differential equation
def dydt(t, y):
    return -y + 2*np.sin(t)

def event(t, y):
    return y[0] - 1

event.terminal = False
event.direction = -1

ts = np.linspace(0, 10, 100)
initial_state = (1, 1)
sol = solve_ivp(dydt, t_span=(0, 10), y0=[10], t_eval=ts, events=event)

t = sol.t
y = sol.y[0]
events = sol.t_events[0]

plt.plot(t, y)
for point in events:
    plt.plot(point, 1, color="green", marker="o", markersize=10)
plt.show()