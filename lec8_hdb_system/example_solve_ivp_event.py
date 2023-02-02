from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

def f(t, r):
    x, y = r
    
    fx = np.cos(y)
    fy = np.sin(x)
    
    return fx, fy

def my_event(t, r):
    x, y = r
    return x - 1

sol = integrate.solve_ivp(
    f, t_span=(0,10), y0=(1,1),
    t_eval=np.linspace(0,10,100),
    events=my_event
)
events = sol.t_events[0]

t = sol.t
x, y = sol.y
plt.plot(t, x)

for point in events:
    plt.plot(point, 1, color="green", marker="o", markersize=10)

plt.axis("scaled")
plt.show()
