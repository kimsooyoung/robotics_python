from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return x**2 - x - 2

# case 1
root = fsolve(func, 3)
# case 2
# root = fsolve(func, 0)

x = np.arange(-6,6,0.1)   # start,stop,step
y = func(x)

plt.figure(1)

plt.plot(x,y)
plt.plot([-6, 6], [0, 0], color="black", linewidth=1)
plt.plot(root, 0, color="green", marker="o", markersize=10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of func")

plt.grid()
plt.show(block=False)
plt.pause(5)
plt.close()
