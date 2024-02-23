import numpy as np
from matplotlib import pyplot as plt

def plot(t, x, u):
    # plt.figure(1, figsize=(8, 8))
    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(t, x[:, 0], color='red', label=r'$x$')
    plt.plot(t, x[:, 1], color='blue', label=r'$\dot{x}$')
    plt.plot(t, x[:, 2], color='green', label=r'$\theta$')
    plt.plot(t, x[:, 3], color='orange', label=r'$\dot{\theta}$')
    plt.xlabel('time')
    plt.ylabel('state')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(u, color='brown', label=r'$u$')
    plt.xlabel('time')
    plt.ylabel('state')
    plt.legend()

    plt.show()

def animate(tspan, x, L, pause=0.01):
    
    W = 0.1

    plt.figure(figsize=(12,4))

    plt.xlim(-6, 6)
    plt.ylim(-1.0, 1.0)
    plt.gca().set_aspect('equal')
    
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Inverted Pendulum')
    
    for i in range(len(tspan)):
        stick, = plt.plot(
            [x[i, 0], x[i, 0] + L*np.sin(x[i, 2])], 
            [0, -L*np.cos(x[i, 2])], 
            'b'
        )
        ball, = plt.plot(
            x[i, 0] + L*np.sin(x[i, 2]), 
            -L*np.cos(x[i, 2]), 
            'ro'
        )
        body, = plt.plot(
            [x[i, 0] - W/2, x[i, 0] + W/2],
            [0, 0],
            linewidth=5,
            color='black'
        )
        
        plt.pause(pause)
        stick.remove()
        ball.remove()
        body.remove()
        
    plt.close()