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
        self.e = 0.9
        self.pause = 0.01
        self.fps = 30
        
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
    contact.direction = -1
    # sol = solve_ivp(projectile,[t0, tf],z0,method='RK45', t_eval=t, dense_output=True, args=(parms.m,parms.g,parms.c))
    sol = solve_ivp(projectile,[t0, tf], z0, method='RK45',
            t_eval=np.linspace(t0, tf, 1001), dense_output=True, 
            events=contact,args=(parms.m,parms.g,parms.c))

    # [xdot, ax, ydot, ay]
    [m,n] = np.shape(sol.y) #4,101
    shape = (n,m) #101,4
    t = sol.t
    z = np.zeros(shape)

    for i in range(0,m):
        z[:,i] = sol.y[i,:]

    z[n-1,3] = -parms.e*z[n-1,3]

    return t,z


def contact(t,z,m,g,c):
    x,xdot,y,ydot = z
    return y

def animate(t,z,parms):
    #interpolation
    t_interp = np.arange(t[0],t[len(t)-1],1/parms.fps)
    [m,n] = np.shape(z)
    shape = (len(t_interp),n)
    z_interp = np.zeros(shape)

    for i in range(0,n-1):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    #plot
    for i in range(0,len(t_interp)):
        prj, =  plt.plot(z_interp[i,0],z_interp[i,2],color='red',marker='o');
        plt.plot([-2, 2],[0, 0],linewidth=2, color='black')

        plt.xlim(min(z[:,0]-1),max(z[:,0]+1))
        plt.ylim(min(z[:,2]-1),max(z[:,2]+1))

        plt.pause(parms.pause)

        if (i != len(t_interp)):
            prj.remove()

    # plt.pause(2)
    plt.close()


parms = parameters()

x0 = 0;
x0dot = 0;
y0 = 2;
y0dot = 10;

t0 = 0;
tend = 10;
z0 = np.array([x0, x0dot, y0, y0dot])
z = z0
t = t0

i = 0
while (t0<=tend):
# for i in range(0,3):
    [t_temp,z_temp] = one_bounce(t0,z0,parms)
    [mm,nn] = np.shape(z_temp)
    if i==0:
        z = np.concatenate(([z], z_temp[1:mm-1,:]), axis=0)
        t = np.concatenate(([t], t_temp[1:mm-1]), axis=0)
    else:
        z = np.concatenate((z, z_temp[1:mm-1,:]), axis=0)
        t = np.concatenate((t, t_temp[1:mm-1]), axis=0)
    x0 = z_temp[mm-1,0]
    x0dot = z_temp[mm-1,1]
    y0 = z_temp[mm-1,2]
    y0dot = z_temp[mm-1,3]
    z0 = np.array([x0, x0dot, y0, y0dot])
    t0 = t_temp[mm-1]
    # print(z0)
    # print(t0)
    i+=1


animate(t,z,parms)

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
