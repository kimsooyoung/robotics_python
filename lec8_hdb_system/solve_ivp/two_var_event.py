import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def f(t, r):
    x, y = r

    fx = np.cos(y)
    fy = np.sin(x)

    return fx, fy


def event(t, r):
    x, y = r
    return x - 1


# event.terminal = False
# event.direction = +1
sol = integrate.solve_ivp(
    f, t_span=(0, 10), y0=(1, 1),
    t_eval=np.linspace(0, 10, 100),
    events=event
)


t = sol.t
x, y = sol.y
events = sol.t_events[0]
print(events)

plt.plot(t, x)
for event in events:
    plt.plot(event, 1, color='green', marker='o', markersize=10)
plt.axis('scaled')
plt.show()
