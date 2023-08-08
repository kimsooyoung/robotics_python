from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import interpolate
from scipy.integrate import odeint
#from mpl_toolkits.mplot3d import art3d
import mpl_toolkits.mplot3d.axes3d as p3

class parameters:
    def __init__(self):
        self.m = 0.468
        self.Ixx = 4.856*1e-3
        self.Iyy = 4.856*1e-3
        self.Izz = 8.801*1e-3
        self.g = 9.81
        self.l = 0.225;
        self.K = 2.980*1e-6;
        self.b = 1.14*1e-7;
        self.Ax = 0.25*0;
        self.Ay = 0.25*0;
        self.Az = 0.25*0;
        self.pause = 0.01
        self.fps = 30

        omega = 1
        speed = 1.075*omega*np.sqrt(1/self.K) #1.075
        dspeed1 = 0.0*speed
        dspeed2 = 0.0*speed
        dspeed3 = 0.0*speed
        dspeed4 = 0.0*speed
        self.omega1 = speed+dspeed1
        self.omega2 = speed+dspeed2
        self.omega3 = speed+dspeed3
        self.omega4 = speed+dspeed4

def cos(angle):
    return np.cos(angle)

def sin(angle):
    return np.sin(angle);

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


def animate(t,Xpos,Xang,parms):
    #interpolation
    Xpos = np.array(Xpos) #convert list to ndarray
    Xang = np.array(Xang)
    t_interp = np.arange(t[0],t[len(t)-1],1/parms.fps)
    [m,n] = np.shape(Xpos)
    shape = (len(t_interp),n)
    Xpos_interp = np.zeros(shape)
    Xang_interp = np.zeros(shape)
    l = parms.l

    for i in range(0,n):
        fpos = interpolate.interp1d(t, Xpos[:,i])
        Xpos_interp[:,i] = fpos(t_interp)
        fang = interpolate.interp1d(t, Xang[:,i])
        Xang_interp[:,i] = fang(t_interp)

    # ll = np.max(np.array([lx,ly,lz]))+0.1
    lmax = np.max(Xpos)
    lmin = np.min(Xpos)
    # print(lmin)
    # print(lmax)

    axle_x = np.array([[-l/2, 0, 0],
                       [l/2, 0, 0]]);
    axle_y = np.array([[0, -l/2,  0],
                      [0, l/2,   0]]);


    [p2,q2] = np.shape(axle_x)

    for ii in range(0,len(t_interp)):
        x = Xpos_interp[ii,0]
        y = Xpos_interp[ii,1]
        z = Xpos_interp[ii,2]
        phi = Xang_interp[ii,0]
        theta = Xang_interp[ii,1]
        psi = Xang_interp[ii,2]
        R = rotation(phi,theta,psi)

        new_axle_x = np.zeros((p2,q2))
        for i in range(0,p2):
            r_body = axle_x[i,:];
            r_world = R.dot(r_body);
            new_axle_x[i,:] = r_world;

        new_axle_x = np.array([x, y, z]) +new_axle_x;
        # print(new_axle_x)

        new_axle_y = np.zeros((p2,q2))
        for i in range(0,p2):
            r_body = axle_y[i,:];
            r_world = R.dot(r_body);
            new_axle_y[i,:] = r_world;

        new_axle_y = np.array([x, y, z]) +new_axle_y;
        # print(new_axle_y)

        ax = p3.Axes3D(fig)
        axle1, = ax.plot(new_axle_x[:,0],new_axle_x[:,1],new_axle_x[:,2], 'ro-', linewidth=3)
        axle2, = ax.plot(new_axle_y[:,0],new_axle_y[:,1],new_axle_y[:,2], 'bo-', linewidth=3)

        ll = 0.2;
        origin = np.array([0,0,-0.5])
        dirn_x = np.array([1, 0, 0]);
        dirn_y = np.array([0, 1, 0]);
        dirn_z = np.array([0, 0, 1]);
        ax.quiver(origin[0],0+origin[1],0+origin[2],dirn_x[0],dirn_x[1],dirn_x[2],
                 length=ll, arrow_length_ratio = .25,normalize=True,color='red')
        ax.quiver(origin[0],origin[1],origin[2],dirn_y[0],dirn_y[1],dirn_y[2],
                 length=ll, arrow_length_ratio = .25,normalize=True,color='green')
        ax.quiver(origin[0],origin[1],origin[2],dirn_z[0],dirn_z[1],dirn_z[2],
                 length=ll, arrow_length_ratio = .25,normalize=True,color='blue')


        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        ax.view_init(azim=-72,elev=20)

        # ax.axis('off');

        plt.pause(parms.pause)

    plt.close()


def eom(X,t,m,Ixx,Iyy,Izz,g,l,K,b,Ax,Ay,Az,omega1,omega2,omega3,omega4):

    i = 0;
    x = X[i]; i +=1;
    y = X[i]; i +=1;
    z = X[i]; i +=1;
    phi = X[i]; i +=1;
    theta = X[i]; i +=1;
    psi = X[i]; i+=1;
    vx = X[i]; i +=1;
    vy = X[i]; i +=1;
    vz = X[i]; i +=1;
    phidot = X[i]; i +=1;
    thetadot = X[i]; i +=1;
    psidot = X[i]; i+=1;

    A = np.zeros((6,6))
    B = np.zeros((6,1))

    A[ 0 , 0 ]= 1.0*m
    A[ 0 , 1 ]= 0
    A[ 0 , 2 ]= 0
    A[ 0 , 3 ]= 0
    A[ 0 , 4 ]= 0
    A[ 0 , 5 ]= 0
    A[ 1 , 0 ]= 0
    A[ 1 , 1 ]= 1.0*m
    A[ 1 , 2 ]= 0
    A[ 1 , 3 ]= 0
    A[ 1 , 4 ]= 0
    A[ 1 , 5 ]= 0
    A[ 2 , 0 ]= 0
    A[ 2 , 1 ]= 0
    A[ 2 , 2 ]= 1.0*m
    A[ 2 , 3 ]= 0
    A[ 2 , 4 ]= 0
    A[ 2 , 5 ]= 0
    A[ 3 , 0 ]= 0
    A[ 3 , 1 ]= 0
    A[ 3 , 2 ]= 0
    A[ 3 , 3 ]= 1.0*Ixx
    A[ 3 , 4 ]= 0
    A[ 3 , 5 ]= -1.0*Ixx*sin(theta)
    A[ 4 , 0 ]= 0
    A[ 4 , 1 ]= 0
    A[ 4 , 2 ]= 0
    A[ 4 , 3 ]= 0
    A[ 4 , 4 ]= 1.0*Iyy*cos(phi)**2 + 1.0*Izz*sin(phi)**2
    A[ 4 , 5 ]= 0.25*(Iyy - Izz)*(sin(2*phi - theta) + sin(2*phi + theta))
    A[ 5 , 0 ]= 0
    A[ 5 , 1 ]= 0
    A[ 5 , 2 ]= 0
    A[ 5 , 3 ]= -1.0*Ixx*sin(theta)
    A[ 5 , 4 ]= 0.25*(Iyy - Izz)*(sin(2*phi - theta) + sin(2*phi + theta))
    A[ 5 , 5 ]= 1.0*Ixx*sin(theta)**2 + 1.0*Iyy*sin(phi)**2*cos(theta)**2 + 1.0*Izz*cos(phi)**2*cos(theta)**2
    B[ 0 ]= -Ax*vx + K*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))*(omega1**2 + omega2**2 + omega3**2 + omega4**2)
    B[ 1 ]= -Ay*vy - K*(sin(phi)*cos(psi) - sin(psi)*sin(theta)*cos(phi))*(omega1**2 + omega2**2 + omega3**2 + omega4**2)
    B[ 2 ]= -Az*vz + K*(omega1**2 + omega2**2 + omega3**2 + omega4**2)*cos(phi)*cos(theta) - g*m
    B[ 3 ]= 1.0*Ixx*psidot*thetadot*cos(theta) + Iyy*(psidot*sin(phi)*cos(theta) + thetadot*cos(phi))*(psidot*cos(phi)*cos(theta) - thetadot*sin(phi)) - 1.0*Izz*(psidot*sin(phi)*cos(theta) + thetadot*cos(phi))*(psidot*cos(phi)*cos(theta) - thetadot*sin(phi)) - K*l*(omega2**2 - omega4**2)
    B[ 4 ]= -1.0*Ixx*phidot*psidot*cos(theta) + 0.5*Ixx*psidot**2*sin(2*theta) - 0.5*Iyy*phidot*psidot*cos(2*phi - theta) - 0.5*Iyy*phidot*psidot*cos(2*phi + theta) + 1.0*Iyy*phidot*thetadot*sin(2*phi) - 0.25*Iyy*psidot**2*sin(2*theta) - 0.125*Iyy*psidot**2*sin(2*phi - 2*theta) + 0.125*Iyy*psidot**2*sin(2*phi + 2*theta) + 0.5*Izz*phidot*psidot*cos(2*phi - theta) + 0.5*Izz*phidot*psidot*cos(2*phi + theta) - 1.0*Izz*phidot*thetadot*sin(2*phi) - 0.25*Izz*psidot**2*sin(2*theta) + 0.125*Izz*psidot**2*sin(2*phi - 2*theta) - 0.125*Izz*psidot**2*sin(2*phi + 2*theta) - 1.0*K*l*omega1**2 + 1.0*K*l*omega3**2
    B[ 5 ]= b*(omega1**2 - omega2**2 + omega3**2 - omega4**2) - 0.5*phidot*(Iyy*psidot*sin(2*phi - theta) + Iyy*psidot*sin(2*phi + theta) + 2*Iyy*thetadot*cos(2*phi) - Izz*psidot*sin(2*phi - theta) - Izz*psidot*sin(2*phi + theta) - 2*Izz*thetadot*cos(2*phi))*cos(theta) + 0.25*thetadot*(4*Ixx*phidot*cos(theta) - 4*Ixx*psidot*sin(2*theta) + 2*Iyy*psidot*sin(2*theta) + Iyy*psidot*sin(2*phi - 2*theta) - Iyy*psidot*sin(2*phi + 2*theta) + Iyy*thetadot*cos(2*phi - theta) - Iyy*thetadot*cos(2*phi + theta) + 2*Izz*psidot*sin(2*theta) - Izz*psidot*sin(2*phi - 2*theta) + Izz*psidot*sin(2*phi + 2*theta) - Izz*thetadot*cos(2*phi - theta) + Izz*thetadot*cos(2*phi + theta))


    invA = np.linalg.inv(A)
    Xddot = invA.dot(B)
    i = 0;
    ax = Xddot[i,0]; i+=1
    ay = Xddot[i,0]; i+=1
    az = Xddot[i,0]; i+=1
    phiddot = Xddot[i,0]; i+=1
    thetaddot = Xddot[i,0]; i+=1
    psiddot = Xddot[i,0]; i+=1

    dXdt = np.array([vx, vy, vz, phidot, thetadot, psidot, ax, ay, az, phiddot,thetaddot,psiddot]);
    return dXdt

parms = parameters()

x0 = 0; y0 = 0; z0 = 0;
vx0 = 0; vy0 = 0; vz0 = 0;
phi0 = 0.0; theta0 = 0; psi0 = 0;
phidot0 = 0; thetadot0 = 0; psidot0 = 0;

t = np.linspace(0, 0.5, 101)
X0 = np.array([x0, y0, z0, phi0, theta0, psi0, vx0, vy0, vz0, phidot0, thetadot0, psidot0])
all_parms = (parms.m,parms.Ixx,parms.Iyy,parms.Izz,parms.g,parms.l,\
             parms.K,parms.b,parms.Ax,parms.Ay,parms.Az,\
             parms.omega1,parms.omega2,parms.omega3,parms.omega4)
X = odeint(eom, X0, t, args=all_parms)

mm = len(X);
x = []; y = []; z = []
phi = []; theta = []; psi = []
vx = []; vy = []; vz = []
phidot = []; thetadot = []; psidot = []
KE = []; PE = []; TE = []
omega_x=[]; omega_y=[]; omega_z = []
omega_body_x=[]; omega_body_y=[]; omega_body_z = []
m = parms.m;
g = parms.g;
Ixx = parms.Ixx
Iyy = parms.Iyy
Izz = parms.Izz
X_pos = []
X_ang = []
for i in range(0,mm):
    j = 0;
    x.append(X[i,j]); j+=1;
    y.append(X[i,j]); j+=1;
    z.append(X[i,j]); j+=1;
    phi.append(X[i,j]); j+=1;
    theta.append(X[i,j]); j+=1;
    psi.append(X[i,j]); j+=1;
    vx.append(X[i,j]); j+=1;
    vy.append(X[i,j]); j+=1;
    vz.append(X[i,j]); j+=1;
    phidot.append(X[i,j]); j+=1;
    thetadot.append(X[i,j]); j+=1;
    psidot.append(X[i,j]); j+=1;
    X_pos.append(np.array([x[i], y[i], z[i]]))
    X_ang.append(np.array([phi[i], theta[i], psi[i]]))

#
#     R_we = np.zeros((3,3))
#     R_we[ 0 , 0 ]= cos(psi[i])*cos(theta[i])
#     R_we[ 0 , 1 ]= -sin(psi[i])
#     R_we[ 0 , 2 ]= 0
#     R_we[ 1 , 0 ]= sin(psi[i])*cos(theta[i])
#     R_we[ 1 , 1 ]= cos(psi[i])
#     R_we[ 1 , 2 ]= 0
#     R_we[ 2 , 0 ]= -sin(theta[i])
#     R_we[ 2 , 1 ]= 0
#     R_we[ 2 , 2 ]= 1
#     rates = np.array([phidot[i], thetadot[i], psidot[i]])
#     omega = R_we.dot(rates);
#     omega_x.append(omega[0])
#     omega_y.append(omega[1])
#     omega_z.append(omega[2])
#
#     R_be = np.zeros((3,3))
#     R_be[ 0 , 0 ]= 1
#     R_be[ 0 , 1 ]= 0
#     R_be[ 0 , 2 ]= -sin(theta[i])
#     R_be[ 1 , 0 ]= 0
#     R_be[ 1 , 1 ]= cos(phi[i])
#     R_be[ 1 , 2 ]= sin(phi[i])*cos(theta[i])
#     R_be[ 2 , 0 ]= 0
#     R_be[ 2 , 1 ]= -sin(phi[i])
#     R_be[ 2 , 2 ]= cos(phi[i])*cos(theta[i])
#     rates = np.array([phidot[i], thetadot[i], psidot[i]])
#     omega_body = R_be.dot(rates);
#     omega_body_x.append(omega_body[0])
#     omega_body_y.append(omega_body[1])
#     omega_body_z.append(omega_body[2])
#
#
# plt.figure(1)
# plt.subplot(2,1,1)
# plt.plot(t,x);
# plt.plot(t,y)
# plt.plot(t,z)
# plt.ylabel('linear position');
# plt.subplot(2,1,2)
# plt.plot(t,vx);
# plt.plot(t,vy)
# plt.plot(t,vz)
# plt.xlabel('time')
# plt.ylabel('linear velocity');

# plt.figure(2)
# plt.subplot(2,1,1)
# plt.plot(t,phi);
# plt.plot(t,theta)
# plt.plot(t,psi)
# plt.ylabel('angular position');
# plt.subplot(2,1,2)
# plt.plot(t,phidot);
# plt.plot(t,thetadot)
# plt.plot(t,psidot)
# plt.xlabel('time')
# plt.ylabel('angular velocity');
# #

#
# ax=plt.figure(4)
# plt.subplot(2,1,1)
# plt.plot(t,omega_x);
# plt.plot(t,omega_y);
# plt.plot(t,omega_z);
# ax.legend(['x', 'y','z'])
# plt.ylabel('omega world');
# plt.subplot(2,1,2)
# plt.plot(t,omega_body_x);
# plt.plot(t,omega_body_y);
# plt.plot(t,omega_body_z);
# ax.legend(['x', 'y','z'])
# plt.ylabel('omega body');
# plt.xlabel('time')
#
#
#plt.show()
# plt.show(block=False)
# plt.pause(2)
# plt.close()
#
fig = plt.figure(5)
animate(t,X_pos,X_ang,parms)
