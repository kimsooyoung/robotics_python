import numpy as np
import matplotlib.pyplot as plt

t0, tf = 0, 1
q0, qf = 10, 20

t = np.linspace(t0, tf, 101)

a0 = q0*tf/(-t0 + tf) - qf*t0/(-t0 + tf)
a1 = -q0/(-t0 + tf) + qf/(-t0 + tf)

q_t = a0 + a1*t
qdot_t = a1 * np.ones( (len(t), 1))

plt.figure(1)

plt.subplot(2, 1, 1)
plt.plot(t, q_t)
plt.ylabel("q")

plt.subplot(2, 1, 2)
plt.plot(t, qdot_t)
plt.ylabel("qdot")

plt.show()