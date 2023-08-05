import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


def free_fall(z, t, g):
    y, y_d = z

    y_dd = -g

    return [y_d, y_dd]


t_0, t_end, N = 0, 3, 100
ts = np.linspace(t_0, t_end, N)

result = odeint(free_fall, [0, 5], ts, args=(9.8,))

plt.plot(ts, result[:, 0], 'b', label='y(t)')
plt.plot(ts, result[:, 1], 'g', label="y\'(t)")

plt.legend(loc='best')
plt.show()
