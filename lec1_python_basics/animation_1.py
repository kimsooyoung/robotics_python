import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 2 * np.pi, 0.1)
y = np.cos(t)

plt.plot(t,y)
for i in range(len(y)):
    temp,  = plt.plot(t[i], y[i], color="green", marker="o", markersize=10)
    plt.pause(0.02)
    temp.remove()

plt.show()
plt.close()