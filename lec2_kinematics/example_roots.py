from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return x**2 - x - 2

root = fsolve(func, 0)
print(root)

x = np.arange(-6,6,0.1)   # start,stop,step
y = func(x)

plt.figure(1)
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of func")
plt.grid()
plt.show(block=False)
plt.pause(5)
plt.close()
