import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

rect = plt.Rectangle((0, 0), 2, 1, color='brown', alpha=1)
circ = plt.Circle((1, 1.5), 0.25, color='yellow')
line = plt.Line2D([0, 2], [1, 1], color='black', linewidth=5)

y1 = np.linspace(0.5, 1.5, 20)
y2 = np.linspace(1.5, 0.5, 20)
y = np.concatenate((y1, y2))

for i in range(len(y)):
    circ = plt.Circle((1, y[i]), 0.25, color='yellow')

    ax.add_patch(circ)
    ax.add_patch(rect)
    ax.add_line(line)

    plt.gca().set_aspect('equal')
    plt.ylim(0, 2)
    plt.xlim(0, 2)

    plt.pause(0.05)
    circ.remove()

plt.close()
