import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import math

def cos(theta):
    return np.cos(theta)

def sin(theta):
    return np.sin(theta)

def rotation(phi,theta,psi):

    R_x = np.array([
                    [1,            0,         0],
                    [0,     cos(phi), -sin(phi)],
                    [0,     sin(phi),  cos(phi)]

                  ])

    R_y = np.array([
                        [cos(theta),  0, sin(theta)],
                        [0,           1,          0],
                        [-sin(theta),  0, cos(theta)]
                      ])

    R_z = np.array( [
                   [cos(psi), -sin(psi), 0],
                   [sin(psi),  cos(psi), 0],
                   [0,            0,         1]
                   ])

    R_temp = np.matmul(R_y,R_x);
    R = np.matmul(R_z,R_temp);
    return R;

def animate(fig_no,phi,theta,psi):
    lx = 0.5;
    ly = 0.25;
    lz = 0.1;
    ll = 1;
    lmax = np.max(np.array([lx, ly, lz,ll]))

    # v0 = np.array([[0,0,0], [lx,0,0], [lx,ly,0], [0,ly,0],
    #               [0,0,lz], [lx,0,lz], [lx,ly,lz], [0,ly,lz]])
    v0 = np.array([[-lx,-ly,-lz], [lx,-ly,-lz], [lx,ly,-lz], [-lx,ly,-lz],
                  [-lx,-ly,lz], [lx,-ly,lz], [lx,ly,lz], [-lx,ly,lz]])

    f = np.array([[0,2,1], [0,3,2], [1,2,6], [1,6,5],
                  [0,5,4], [0,1,5], [4,5,6], [6,7,4],
                  [3,7,6], [6,2,3], [0,4,7], [7,3,0]])

    v1 = np.zeros(np.shape(v0))
    [m,n] = np.shape(v1)
    R = rotation(phi,theta,psi)
    for i in range(0,m):
        vec = np.array([v0[i,0], v0[i,1], v0[i,2]])
        vec = R.dot(vec)
        v1[i,0] = vec[0];
        v1[i,1] = vec[1];
        v1[i,2] = vec[2];


    fig = plt.figure(1)
    ax = fig.add_subplot(2, 2, fig_no ,projection="3d")
    pc0 = art3d.Poly3DCollection(v0[f], facecolors="lightblue",alpha=0.5) #, edgecolor="black")
    pc1 = art3d.Poly3DCollection(v1[f], facecolors="blue",alpha=0.25) #, edgecolor="black")

    # ax.add_collection(pc0)
    ax.add_collection(pc1)

    origin = np.array([0,0,0])
    dirn_x = np.array([1, 0, 0]); dirn_x = R.dot(dirn_x);
    dirn_y = np.array([0, 1, 0]); dirn_y = R.dot(dirn_y);
    dirn_z = np.array([0, 0, 1]); dirn_z = R.dot(dirn_z);
    ax.quiver(origin[0],origin[1],origin[2],dirn_x[0],dirn_x[1],dirn_x[2],
             length=1, arrow_length_ratio = .1,normalize=True,color='red')
    ax.quiver(origin[0],origin[1],origin[2],dirn_y[0],dirn_y[1],dirn_y[2],
             length=1, arrow_length_ratio = .1,normalize=True,color='green')
    ax.quiver(origin[0],origin[1],origin[2],dirn_z[0],dirn_z[1],dirn_z[2],
             length=1, arrow_length_ratio = .1,normalize=True,color='blue')

    fac = 180/np.pi
    phideg = math.trunc(float(phi*fac))
    thetadeg = math.trunc(float(theta*fac))
    psideg = math.trunc(float(psi*fac))
    subtit = 'phi='+str(phideg)+';'+'theta='+str(thetadeg)+';'+'psi='+str(psideg)+';'
    ax.set_title(subtit)
    ax.set_xlim(-lmax,lmax)
    ax.set_ylim(-lmax,lmax)
    ax.set_zlim(-lmax,lmax)
    ax.axis('off');

    #plt.show()
    # plt.show(block=False)
    # plt.pause(5)
    # plt.close()


phi = 0;
theta = 0;
psi = 0;
animate(1,phi,theta,psi)

phi = 0;
theta = 0;
psi = np.pi/2;
animate(2,phi,theta,psi)

phi = 0;
theta = np.pi/2;
psi = np.pi/2
animate(3,phi,theta,psi)

phi = np.pi/2;
theta = np.pi/2;
psi = np.pi/2
animate(4,phi,theta,psi)

plt.show()
# plt.show(block=False)
# plt.pause(10)
# plt.close()
