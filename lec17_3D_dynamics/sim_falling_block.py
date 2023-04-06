from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import interpolate
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import art3d

class parameters:
    def __init__(self):
        self.m = 10
        
        self.lx = 0.1
        self.ly = 0.05
        self.lz = 0.02
        
        self.Ixx = self.m/12*(self.ly**2+self.lz**2)
        self.Iyy = self.m/12*(self.lx**2+self.lz**2)
        self.Izz = self.m/12*(self.lx**2+self.ly**2)
        
        self.g = 9.8
        
        self.pause = 0.05
        self.fps = 50

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

    return R_z @ R_y @ R_x

def animate(t,Xpos,Xang,parms):
    #interpolation
    Xpos = np.array(Xpos) #convert list to ndarray
    Xang = np.array(Xang)
    t_interp = np.arange(t[0],t[len(t)-1],1/parms.fps)
    [m,n] = np.shape(Xpos)
    shape = (len(t_interp),n)
    Xpos_interp = np.zeros(shape)
    Xang_interp = np.zeros(shape)

    for i in range(0,n):
        fpos = interpolate.interp1d(t, Xpos[:,i])
        Xpos_interp[:,i] = fpos(t_interp)
        fang = interpolate.interp1d(t, Xang[:,i])
        Xang_interp[:,i] = fang(t_interp)

    lx, ly, lz = parms.lx, parms.ly, parms.lz
    ll = np.max(np.array([lx,ly,lz])) + 0.1
    
    lmax, lmin = np.max(Xpos), np.min(Xpos)
    
    #plot
    v0 =  np.array([[-lx,-ly,-lz], [lx,-ly,-lz], [lx,ly,-lz], [-lx,ly,-lz],
                  [-lx,-ly,lz], [lx,-ly,lz], [lx,ly,lz], [-lx,ly,lz]])

    f = np.array([[0,2,1], [0,3,2], [1,2,6], [1,6,5],
                  [0,5,4], [0,1,5], [4,5,6], [6,7,4],
                  [3,7,6], [6,2,3], [0,4,7], [7,3,0]])

    for i in range(0,len(t_interp)):
        x = Xpos_interp[i,0]
        y = Xpos_interp[i,1]
        z = Xpos_interp[i,2]
        phi = Xang_interp[i,0]
        theta = Xang_interp[i,1]
        psi = Xang_interp[i,2]

        v1 = np.zeros(np.shape(v0))
        [m,n] = np.shape(v1)
        R = rotation(phi,theta,psi)
        for i in range(0,m):
            vec = np.array([v0[i,0], v0[i,1], v0[i,2]])
            vec = R.dot(vec)
            v1[i,0] = vec[0]+x;
            v1[i,1] = vec[1]+y;
            v1[i,2] = vec[2]+z;

        fig = plt.figure(1)
        ax = fig.add_subplot(projection="3d")
        # pc0 = art3d.Poly3DCollection(v0[f], facecolors="lightblue",alpha=0.5) #, edgecolor="black")
        pc1 = art3d.Poly3DCollection(v1[f], facecolors="blue",alpha=0.25) #, edgecolor="black")

        # ax.add_collection(pc0)
        ax.add_collection(pc1)
        
        origin = np.array([0,0,0])
        dirn_x = np.array([1, 0, 0]); dirn_x = R.dot(dirn_x);
        dirn_y = np.array([0, 1, 0]); dirn_y = R.dot(dirn_y);
        dirn_z = np.array([0, 0, 1]); dirn_z = R.dot(dirn_z);
        ax.quiver(x+origin[0],y+origin[1],z+origin[2],dirn_x[0],dirn_x[1],dirn_x[2],
                 length=ll, arrow_length_ratio = .1,normalize=True,color='red')
        ax.quiver(x+origin[0],y+origin[1],z+origin[2],dirn_y[0],dirn_y[1],dirn_y[2],
                 length=ll, arrow_length_ratio = .1,normalize=True,color='green')
        ax.quiver(x+origin[0],y+origin[1],z+origin[2],dirn_z[0],dirn_z[1],dirn_z[2],
                 length=ll, arrow_length_ratio = .1,normalize=True,color='blue')

        ax.set_xlim(lmin,lmax)
        ax.set_ylim(lmin,lmax)
        ax.set_zlim(lmin,lmax)
        ax.axis('off');

        plt.pause(parms.pause)

    plt.close()

def eom(X,t ,m,Ixx,Iyy,Izz,g):

    x,y,z,phi,theta,psi,vx,vy,vz,phidot,thetadot,psidot = X

    A = np.zeros((6,6))
    b = np.zeros((6,1))

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
    b[ 0 ]= 0
    b[ 1 ]= 0
    b[ 2 ]= -g*m
    b[ 3 ]= 1.0*Ixx*psidot*thetadot*cos(theta) + Iyy*(psidot*sin(phi)*cos(theta) + thetadot*cos(phi))*(psidot*cos(phi)*cos(theta) - thetadot*sin(phi)) - 1.0*Izz*(psidot*sin(phi)*cos(theta) + thetadot*cos(phi))*(psidot*cos(phi)*cos(theta) - thetadot*sin(phi))
    b[ 4 ]= -1.0*Ixx*phidot*psidot*cos(theta) + 0.5*Ixx*psidot**2*sin(2*theta) - 0.5*Iyy*phidot*psidot*cos(2*phi - theta) - 0.5*Iyy*phidot*psidot*cos(2*phi + theta) + 1.0*Iyy*phidot*thetadot*sin(2*phi) - 0.25*Iyy*psidot**2*sin(2*theta) - 0.125*Iyy*psidot**2*sin(2*phi - 2*theta) + 0.125*Iyy*psidot**2*sin(2*phi + 2*theta) + 0.5*Izz*phidot*psidot*cos(2*phi - theta) + 0.5*Izz*phidot*psidot*cos(2*phi + theta) - 1.0*Izz*phidot*thetadot*sin(2*phi) - 0.25*Izz*psidot**2*sin(2*theta) + 0.125*Izz*psidot**2*sin(2*phi - 2*theta) - 0.125*Izz*psidot**2*sin(2*phi + 2*theta)
    b[ 5 ]= -0.5*phidot*(Iyy*psidot*sin(2*phi - theta) + Iyy*psidot*sin(2*phi + theta) + 2*Iyy*thetadot*cos(2*phi) - Izz*psidot*sin(2*phi - theta) - Izz*psidot*sin(2*phi + theta) - 2*Izz*thetadot*cos(2*phi))*cos(theta) + 0.25*thetadot*(4*Ixx*phidot*cos(theta) - 4*Ixx*psidot*sin(2*theta) + 2*Iyy*psidot*sin(2*theta) + Iyy*psidot*sin(2*phi - 2*theta) - Iyy*psidot*sin(2*phi + 2*theta) + Iyy*thetadot*cos(2*phi - theta) - Iyy*thetadot*cos(2*phi + theta) + 2*Izz*psidot*sin(2*theta) - Izz*psidot*sin(2*phi - 2*theta) + Izz*psidot*sin(2*phi + 2*theta) - Izz*thetadot*cos(2*phi - theta) + Izz*thetadot*cos(2*phi + theta))

    invA = np.linalg.inv(A)
    Xddot = invA.dot(b)
    
    ax, ay, az, phiddot, thetaddot, psiddot = Xddot[:,0]
    dXdt = np.array([vx, vy, vz, phidot, thetadot, psidot, ax, ay, az, phiddot,thetaddot,psiddot]);
    return dXdt


def plot(ts, x, y, z, vx, vy, vz, phi, theta, psi, phidot, thetadot, psidot, KE, PE, TE, omega_x, omega_y, omega_z, omega_body_x, omega_body_y, omega_body_z):
    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(ts,x);
    plt.plot(ts,y)
    plt.plot(ts,z)
    plt.ylabel('linear position');
    plt.subplot(2,1,2)
    plt.plot(ts,vx);
    plt.plot(ts,vy)
    plt.plot(ts,vz)
    plt.xlabel('time')
    plt.ylabel('linear velocity');

    plt.figure(3)
    plt.subplot(2,1,1)
    plt.plot(ts,phi);
    plt.plot(ts,theta)
    plt.plot(ts,psi)
    plt.ylabel('angular position');
    plt.subplot(2,1,2)
    plt.plot(ts,phidot);
    plt.plot(ts,thetadot)
    plt.plot(ts,psidot)
    plt.xlabel('time')
    plt.ylabel('angular velocity');

    ax=plt.figure(4)
    plt.plot(ts,PE,'b-.')
    plt.plot(ts,KE,'r:')
    plt.plot(ts,TE,'k')
    plt.xlabel('time')
    plt.ylabel('energy');
    ax.legend(['PE', 'KE','TE'])

    ax=plt.figure(5)
    plt.subplot(2,1,1)
    plt.plot(ts,omega_x);
    plt.plot(ts,omega_y);
    plt.plot(ts,omega_z);
    ax.legend(['x', 'y','z'])
    plt.ylabel('omega world');
    plt.subplot(2,1,2)
    plt.plot(ts,omega_body_x);
    plt.plot(ts,omega_body_y);
    plt.plot(ts,omega_body_z);
    ax.legend(['x', 'y','z'])
    plt.ylabel('omega body');
    plt.xlabel('time')

    plt.show()
    

if __name__=='__main__':

    parms = parameters()

    # initial conditions
    x0, y0, z0, phi0, theta0, psi0 = 0, 0, 0, 0, 0, 0
    vx0, vy0, vz0, phidot0, thetadot0, psidot0 = 0, 0, 5, 3, -4, 5

    t0, tend, N = 0, 1, 100
    ts = np.linspace(0, 1, N)
    X0 = np.array([x0, y0, z0, phi0, theta0, psi0, vx0, vy0, vz0, phidot0, thetadot0, psidot0])
    
    m, Ixx, Iyy, Izz, g = parms.m, parms.Ixx, parms.Iyy, parms.Izz, parms.g
    all_parms = (m, Ixx, Iyy, Izz, g)
    
    X = odeint(eom, X0, ts, args=all_parms)
    
    x = []; y = []; z = []
    phi = []; theta = []; psi = []
    
    vx = []; vy = []; vz = []
    phidot = []; thetadot = []; psidot = []
    
    KE = []; PE = []; TE = []
    
    omega_x=[]; omega_y=[]; omega_z = []
    omega_body_x=[]; omega_body_y=[]; omega_body_z = []
    
    X_pos = []
    X_ang = []
    for i in range(N):
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
        KE_temp= 0.5*Ixx*(phidot[i] - psidot[i]*sin(theta[i]))**2 + \
                0.5*Iyy*(psidot[i]*sin(phi[i])*cos(theta[i]) + thetadot[i]*cos(phi[i]))**2 + \
                0.5*Izz*(psidot[i]*cos(phi[i])*cos(theta[i]) - thetadot[i]*sin(phi[i]))**2 + \
                0.5*m*(vx[i]**2 + vy[i]**2 + vz[i]**2)
        PE_temp= g*m*z[i]
        PE.append(PE_temp);
        KE.append(KE_temp);
        TE.append(PE_temp+KE_temp)
        X_pos.append(np.array([x[i], y[i], z[i]]))
        X_ang.append(np.array([phi[i], theta[i], psi[i]]))

        R_we = np.zeros((3,3))
        R_we[ 0 , 0 ]= cos(psi[i])*cos(theta[i])
        R_we[ 0 , 1 ]= -sin(psi[i])
        R_we[ 0 , 2 ]= 0
        R_we[ 1 , 0 ]= sin(psi[i])*cos(theta[i])
        R_we[ 1 , 1 ]= cos(psi[i])
        R_we[ 1 , 2 ]= 0
        R_we[ 2 , 0 ]= -sin(theta[i])
        R_we[ 2 , 1 ]= 0
        R_we[ 2 , 2 ]= 1
        
        rates = np.array([phidot[i], thetadot[i], psidot[i]])
        omega = R_we.dot(rates);
        omega_x.append(omega[0])
        omega_y.append(omega[1])
        omega_z.append(omega[2])

        R_be = np.zeros((3,3))
        R_be[ 0 , 0 ]= 1
        R_be[ 0 , 1 ]= 0
        R_be[ 0 , 2 ]= -sin(theta[i])
        R_be[ 1 , 0 ]= 0
        R_be[ 1 , 1 ]= cos(phi[i])
        R_be[ 1 , 2 ]= sin(phi[i])*cos(theta[i])
        R_be[ 2 , 0 ]= 0
        R_be[ 2 , 1 ]= -sin(phi[i])
        R_be[ 2 , 2 ]= cos(phi[i])*cos(theta[i])
        rates = np.array([phidot[i], thetadot[i], psidot[i]])
        omega_body = R_be.dot(rates);
        omega_body_x.append(omega_body[0])
        omega_body_y.append(omega_body[1])
        omega_body_z.append(omega_body[2])

    animate(ts, X_pos, X_ang, parms)
    plot(ts, x, y, z, vx, vy, vz, phi, theta, psi, phidot, thetadot, psidot, KE, PE, TE, omega_x, omega_y, omega_z, omega_body_x, omega_body_y, omega_body_z)