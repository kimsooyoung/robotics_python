import numpy as np 
import matplotlib.pyplot as plt

def cos(angle): 
    return np.cos(angle) 

def sin(angle): 
    return np.sin(angle) 

def nlink_plot(t, z, params): 

    plt.figure(1)
    plt.subplot(2,1,1)

    plt.plot(t,z[:,0],'r--', label=r'$\theta_1$')
    plt.plot(t,z[:,2],'b', label=r'$\theta_2$')
    plt.plot(t,z[:,4],'g', label=r'$\theta_3$')
    plt.ylabel("position")
    plt.legend(loc=(1.0, 1.0), ncol=1, fontsize=7)
    
    plt.subplot(2,1,2)
    plt.plot(t,z[:,1],'r--', label=r'$\dot{\theta_1}$')
    plt.plot(t,z[:,3],'b', label=r'$\dot{\theta_2}$')
    plt.plot(t,z[:,5],'g', label=r'$\dot{\theta_3}$')
    plt.ylabel("velocity")
    plt.legend(loc=(1.0, 1.0), ncol=1, fontsize=7)
    plt.xlabel('time')

    plt.show()