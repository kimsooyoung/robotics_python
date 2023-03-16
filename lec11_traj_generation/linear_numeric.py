import numpy as np
import matplotlib.pyplot as plt

q0 = 0;
qf = 1;
t0 = 0;
tf = 1;

a0 = q0*tf/(-t0 + tf) - qf*t0/(-t0 + tf);
a1 = -q0/(-t0 + tf) + qf/(-t0 + tf);

t = np.linspace(t0, tf, 101)
q = a0+a1*t;
qdot = a1*np.ones( (len(t),1))

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(t,q)
plt.ylabel("q");
plt.subplot(2, 1, 2)
plt.plot(t,qdot)
plt.ylabel('qdot');
plt.show()




# print(np.shape(t))
# print(np.shape(q))
# print(np.shape(qdot))
