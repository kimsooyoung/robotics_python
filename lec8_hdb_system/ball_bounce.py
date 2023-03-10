from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import interpolate
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

class parameters:
    def __init__(self):
        self.g = 9.81
        self.m = 1
        self.c = 0
        self.e = 0.8
        self.pause = 0.005
        self.fps = 10
        
def projectile(t, z, m,g,c):

    x,xdot,y,ydot = z
    v = np.sqrt(xdot**2+ydot**2);

    #%%%% drag is prop to v^2
    dragX = c*v*xdot;
    dragY = c*v*ydot;

    #%%%% net acceleration %%%
    ax = 0-(dragX/m);
    ay = -g-(dragY/m);

    return [xdot, ax, ydot, ay]

# t0 : initial time
# z0 : initial state
def one_bounce(t0,z0,parms):

    tf = t0 + 5
    contact.terminal = True
    
    # 위에서 아래로 내려갈 때만 잡는다.
    contact.direction = -1

    sol = solve_ivp(projectile, t_span=(t0, tf), y0=z0, method='RK45',
        t_eval=np.linspace(t0, tf, 1001), dense_output=True, 
        events=contact, args=(parms.m,parms.g,parms.c)
    )

    # [xdot, ax, ydot, ay]
    [m,n] = np.shape(sol.y) #4,101
    shape = (n,m) #101,4
    t = sol.t
    z = np.zeros(shape)

    for i in range(0,m):
        z[:,i] = sol.y[i,:]

    # z[n-1,3] = -parms.e*z[n-1,3]
    z[n-1,3] *= -parms.e

    return t,z

# return값이 0일 때가 조건이 되어 해당 point가 반환된다.
def contact(t,z,m,g,c):
    x,xdot,y,ydot = z
    return y

def simulation(t0, z0):

    t = np.array([t0])
    z = np.zeros((1, 4))

    # bouncing iteration 
    i = 0
    while (t0 <= tend):
        [t_temp, z_temp] = one_bounce(t0, z0, parms)

        z = np.concatenate((z, z_temp), axis=0)
        t = np.concatenate((t, t_temp), axis=0)

        z0 = z_temp[-1]
        t0 = t_temp[-1]
        # print(z0)
        # print(t0)
        i+=1

    return t, z


def animate(t,z,parms):
    #interpolation
    t_interp = np.arange(t[0],t[len(t)-1],1/parms.fps)
    [m,n] = np.shape(z)
    shape = (len(t_interp),n)
    z_interp = np.zeros(shape)

    for i in range(0,n-1):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    for i in range(0,len(t_interp)):
        prj, =  plt.plot(z_interp[i,0],z_interp[i,2],color='red',marker='o');
        plt.plot([-2, 2],[0, 0],linewidth=2, color='black')

        plt.xlim(min(z[:,0]-1),max(z[:,0]+1))
        plt.ylim(min(z[:,2]-1),max(z[:,2]+1))

        plt.pause(parms.pause)

        if (i != len(t_interp)):
            prj.remove()

    plt.close()

def plot_result(t, z):
    
    plt.figure(1)
    
    plt.subplot(2,1,1)
    plt.plot(t,z[:,2],'r')
    plt.ylabel('y')
    
    plt.subplot(2,1,2)
    plt.plot(t,z[:,3],'r')
    plt.ylabel('ydot')
    plt.xlabel('time')
    
    plt.show(block=False)
    plt.pause(3)
    plt.close()

if __name__=="__main__":

    parms = parameters()

    # define initial cond
    # 초기 위로 올라가는 속도 10
    x0, x0dot, y0, y0dot = (0, 0, 2, 10)
    t0, tend = (0, 10)

    # initial state
    z0 = np.array([x0, x0dot, y0, y0dot])

    t, z = simulation(t0, z0)
    animate(t, z, parms)
    # plot_result(t, z)