import numpy as np
import matplotlib.pyplot as plt

q0 = 0;
qf = 1;
t0 = 0;
tf = 1;

a0 = q0*(3*t0*tf**2 - tf**3)/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3) + qf*(t0**3 - 3*t0**2*tf)/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3)
a1 = -6*q0*t0*tf/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3) + 6*qf*t0*tf/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3)
a2 = q0*(3*t0 + 3*tf)/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3) + qf*(-3*t0 - 3*tf)/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3)
a3 = -2*q0/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3) + 2*qf/(t0**3 - 3*t0**2*tf + 3*t0*tf**2 - tf**3)

t = np.linspace(t0, tf, 101)
q = a0+a1*t+a2*t**2+a3*t**3;
qdot = a1+2*a2*t+3*a3*t**2;
qddot = 2*a2+6*a3*t;

plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(t,q)
plt.ylabel("q");
plt.subplot(3, 1, 2)
plt.plot(t,qdot)
plt.ylabel('qdot');
plt.subplot(3, 1, 3)
plt.plot(t,qddot)
plt.ylabel('qddot');
plt.show()
