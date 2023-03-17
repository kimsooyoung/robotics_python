import numpy as np
import matplotlib.pyplot as plt

t0, tf = 0, 1
q0, qf = 10, 20

t = np.linspace(t0, tf, 101)


a0 = q0*(3*t0*tf**2 - tf**3)/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3) + qf*(t0**3 - 3*t0**2*tf)/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3)
a1 = -6*q0*t0*tf/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3) + 6*qf*t0*tf/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3)
a2 = q0*(3*t0 + 3*tf)/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3) - qf*(3*t0 + 3*tf)/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3)
a3 = -2*q0/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3) + 2*qf/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3)

q_t = a0 + a1*t + a2*t**2 + a3*t**3
qdot_t = a1 + 2*a2*t + 3*a3*t**2
qdotdot_t = 2*a2 + 6*a3*t

plt.figure(1)

plt.subplot(3, 1, 1)
plt.plot(t, q_t)
plt.ylabel("q")

plt.subplot(3, 1, 2)
plt.plot(t, qdot_t)
plt.ylabel("qdot")

plt.subplot(3, 1, 3)
plt.plot(t, qdotdot_t)
plt.ylabel("qdotdot")

plt.show()