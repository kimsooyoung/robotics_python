from matplotlib import pyplot as plt
import math
from scipy import interpolate
import numpy as np

class parameters:
    def __init__(self):
        self.px = 0.05
        self.py = 0
        self.Kp = 100;
        self.R = 0.1
        self.pause = 0.01
        self.fps =10

def ptP_to_ptC(x_p,y_p,theta,parms):
    cos = np.cos(theta)
    sin = np.sin(theta)
    R = np.array([[cos, -sin],
                  [sin, cos]])
    r = np.array([parms.px,parms.py])
    p = np.array([x_p,y_p])
    c = -np.matmul(R,np.transpose(r)) + np.transpose(p)
    return c

def ptC_to_ptP(x_c,y_c,theta,parms):
    cos = np.cos(theta)
    sin = np.sin(theta)
    R = np.array([[cos, -sin],
                  [sin, cos]])
    r = np.array([parms.px,parms.py])
    c = np.array([x_c,y_c])
    p = np.matmul(R,np.transpose(r)) + np.transpose(c)
    return p


def animate(t,z,p,parms):
    R = parms.R;
    phi = np.arange(0,2*np.pi,0.25)
    x_circle = R*np.cos(phi)
    y_circle = R*np.sin(phi)

    for i in range(0,len(t)):
        x = z[i,0]
        y = z[i,1]
        theta = z[i,2]

        x_robot = x + x_circle
        y_robot = y + y_circle

        x2 = x + R*np.cos(theta)
        y2 = y + R*np.sin(theta)

        line, = plt.plot([x, x2],[y, y2],color="black")
        robot,  = plt.plot(x_robot,y_robot,color='black')
        shape, = plt.plot(p[0:i,0],p[0:i,1],color='red');

        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.gca().set_aspect('equal')
        plt.pause(parms.pause)
        line.remove()
        robot.remove()
        shape.remove()

plt.close()


def euler_integration(tspan,z0,u):
    v = u[0]
    omega = u[1]
    h = t[1]-t[0]

    x0 = z0[0]
    y0 = z0[1]
    theta0 = z0[2]

    xdot_c = v*math.cos(theta0)
    ydot_c = v*math.sin(theta0)
    thetadot = omega

    x1 = x0 + xdot_c*h
    y1 = y0 + ydot_c*h
    theta1 = theta0 + thetadot*h

    z1 = [x1, y1, theta1]
    return z1

# %%%%%% parm is struct that helps pass many arguments %%
parms = parameters()

t0 = 0;
tend = 10;
t = np.arange(t0,tend,0.01 )

# %%% circle %%%
# x_ref = 1*np.cos(2*math.pi*t/tend); y_ref = 1*np.sin(2*math.pi*t/tend);
# plt.plot(x_ref,y_ref)
# plt.show()

# % %%% astroid (note these are coordinates of point P) %%
x_center = 0;
y_center = 0;
a = 1;
x_ref = x_center+a*np.cos(2*math.pi*t/tend)**3
y_ref = y_center+a*np.sin(2*math.pi*t/tend)**3
# plt.plot(x_ref,y_ref)
# plt.gca().set_aspect('equal')
# plt.show()

theta0 = math.pi/2; #guessed to be zero. Can be made better by specifying a better value based on the specific curve
c = ptP_to_ptC(x_ref[0],y_ref[0],theta0,parms); #given the x_ref(1) and y_ref(1) (position of E at start) this step computes the correct starting position for the car (i.e., point P)
x0 = c[0]
y0 = c[1]
z0 = np.array([x0, y0, theta0]); #x, y, theta at the start for point C
z = np.array(z0);
p0 = [x_ref[0],y_ref[0]]
p = np.array(p0);
e = np.array([0,0]) ;
v = [];
omega = [];

for i in range(0,len(t)-1):
    # 1. get x_c,y_c position
    x_c = z0[0];
    y_c = z0[1];
    theta = z0[2];

    # 2. get x_p, y_p from x_c,y_c
    p0 = ptC_to_ptP(z0[0],z0[1],z0[2],parms);
    x_p = p0[0]
    y_p = p0[1]

    #% 3. get error = xe-x_ref and ye-y_ref
    error = [x_ref[i+1]-x_p, y_ref[i+1]-y_p];
    e = np.vstack([e, error])

    # %4. get u = [v, omega] from the errors
    b = [parms.Kp*error[0], parms.Kp*error[1]];
    cos = np.cos(theta);
    sin = np.sin(theta);

    px = parms.px;
    py = parms.py;
    Ainv = np.array([[cos-(py/px)*sin, sin+(py/px)*cos],
                 [-(1/px)*sin,     (1/px)*cos]    ]);
    u = np.matmul(Ainv,np.transpose(b))
    #u = Ainv*b; #%u = [v omega]
    v.append(u[0])
    omega.append(u[1])

    #     % 5. now control the car based on u = [v omega]
    z0 = euler_integration([t[i], t[i+1]],z0,[u[0],u[1]])
    z = np.vstack([z, z0])
    p0 = ptC_to_ptP(z0[0],z0[1],z0[2],parms);
    p = np.vstack([p, p0])

#interpolation
t_interp = np.arange(t0,tend,1/parms.fps)
f_z1 = interpolate.interp1d(t, z[:,0])
f_z2 = interpolate.interp1d(t, z[:,1])
f_z3 = interpolate.interp1d(t, z[:,2])
shape = (len(t_interp),3)
z_interp = np.zeros(shape)
z_interp[:,0] = f_z1(t_interp)
z_interp[:,1] = f_z2(t_interp)
z_interp[:,2] = f_z3(t_interp)

f_p1 = interpolate.interp1d(t, p[:,0])
f_p2 = interpolate.interp1d(t, p[:,1])
shape = (len(t_interp),2)
p_interp = np.zeros(shape)
p_interp[:,0] = f_p1(t_interp)
p_interp[:,1] = f_p2(t_interp)

animate(t_interp,z_interp,p_interp,parms)
