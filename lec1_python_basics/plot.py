import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 6, 50)
y = np.sin(t)

plt.figure(1)
# b => blue, o => circle
plt.plot(t,y,'bo')
plt.xlabel("t")
plt.ylabel("sin(t)")
plt.show(block=False)
plt.pause(5)
plt.close()
