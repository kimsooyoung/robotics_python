# Copyright 2022 @RoadBalance
# Reference from https://pab47.github.io/legs.html
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from matplotlib import pyplot as plt
import numpy as np
import math

from scipy import interpolate
from scipy.integrate import odeint
from cython_dynamics import projectile_cython

class parameters:
    def __init__(self):
        self.g = 9.81
        self.m = 1
        self.c = 0.47
        self.pause = 0.01
        self.fps = 10
        self.t_length = 5

def interpolation(t, z, params):

    #interpolation
    t_interp = np.arange(t[0], t[len(t)-1], 1/params.fps)
    [rows, cols] = np.shape(z)
    z_interp = np.zeros((len(t_interp), cols))

    for i in range(0, cols):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    return t_interp, z_interp

def animate(t_interp,z_interp,parms):
    #plot
    for i in range(0,len(t_interp)):
        traj, = plt.plot(z_interp[0:i,0],z_interp[0:i,2],color='red');
        prj, =  plt.plot(z_interp[i,0],z_interp[i,2],color='red',marker='o');

        plt.xlim(min(z[:,0]-1),max(z[:,0]+1))
        plt.ylim(min(z[:,2]-1),max(z[:,2]+1))
        # plt.gca().set_aspect('equal')
        plt.pause(parms.pause)
        traj.remove()
        prj.remove()

    # plt.close()
    fig2, (ax, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)

    ax.set_title("X")
    ax.plot(t_interp, z_interp[:,0], color="green")

    ax2.set_title("X dot")
    ax2.plot(t_interp, z_interp[:,1], color="orange")

    ax3.set_title("Y")
    ax3.plot(t_interp, z_interp[:,2], color="green")

    ax4.set_title("Y dot")
    ax4.plot(t_interp, z_interp[:,3], color="orange")

    plt.show()

if __name__=="__main__":
    params = parameters()
    # initial state
    x0, x0dot, y0, y0dot = (0, 100, 0, 100*math.tan(math.pi/3))
    z0 = np.array([x0, x0dot, y0, y0dot])

    t_start, t_end = (0, params.t_length)
    t = np.arange(t_start, t_end, 0.01)

    try:
        # calc states from ode solved
        import time
        
        start = time.time()
        z = odeint(projectile_cython.projectile, z0, t, args=(params.m,params.g,params.c))
        end = time.time()
        print(f"{end - start:.5f} sec") # 0.00141
    except Exception as e:
        print(e)
    finally:
        # interpolation for ploting
        t_interp, z_interp = interpolation(t, z, params)
        # Draw plot
        animate(t_interp,z_interp,params)
        print("Everything done!")

