from matplotlib import pyplot as plt
import math
from scipy import interpolate
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

class parameters:
    def __init__(self):
        # 목적지에 도달해야 하는 offset
        # 목적지와 딱 맞게 가고 싶다면 px=0.0이어야 하는건가?
        self.px = 0.01
        self.py = 0
        self.Kp = 100;
        # 로봇 반지름
        self.R = 0.1
        self.pause = 0.1
        self.fps =10
        self.t_length = 10

def ptP_to_ptC(x_p,y_p,theta,params):
    # x_p, y_p : 로봇이 도달해야 할 좌표
    # params.px,params.py : 로봇이 회전 후 가져야 할 offset
    # 실제 해당 offset 만큼은 여유를 둬야 하므로 -np.matmul이 되었다.
    cos = np.cos(theta)
    sin = np.sin(theta)
    R = np.array([[cos, -sin],
                  [sin, cos]])
    r = np.array([params.px,params.py])
    p = np.array([x_p,y_p])
    c = -np.matmul(R,np.transpose(r)) + np.transpose(p)
    return c

def ptC_to_ptP(x_c,y_c,theta,params):
    cos = np.cos(theta)
    sin = np.sin(theta)
    R = np.array([[cos, -sin],
                  [sin, cos]])
    r = np.array([params.px,params.py])
    c = np.array([x_c,y_c])
    p = np.matmul(R,np.transpose(r)) + np.transpose(c)
    return p


def animate(t, z, p, params):
    R = params.R;
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

        # 로봇 방향을 나타내는 작대기
        line, = plt.plot([x, x2],[y, y2],color="black")
        robot,  = plt.plot(x_robot,y_robot,color='black')
        # 로봇이 그리는 경로
        shape, = plt.plot(p[0:i,0],p[0:i,1],color='red');

        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.gca().set_aspect('equal')
        plt.pause(params.pause)
        line.remove()
        robot.remove()
        shape.remove()

plt.close()


def euler_integration(tspan,z0,u):
    v = u[0]
    omega = u[1]
    h = tspan[1]-tspan[0]

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


def generate_path(params=None, path_type="astroid", show_path=False):
    t0 = 0;
    tend = params.t_length;
    t = np.arange(t0,tend,0.01)

    if path_type == "circle":
        R = 1.0
        x_ref = R * np.cos(2 * math.pi * t/tend)
        y_ref = R * np.sin(2 * math.pi * t/tend)
    # generate astroid-shape path 
    # (note these are coordinates of point P)
    elif path_type == "astroid":
        x_center = 0;
        y_center = 0;
        a = 1;
        x_ref = x_center + a * np.cos(2 * math.pi * t/tend) **3
        y_ref = y_center + a * np.sin(2 * math.pi * t/tend) **3
        
    if show_path == True:
        plt.plot(x_ref,y_ref)
        plt.gca().set_aspect('equal')
        plt.show()
        plt.pause(1.0)
        plt.close()

    return x_ref, y_ref


def motion_simulation(params, path=None):
    # guessed to be zero. Can be made better by 
    # specifying a better value based on the specific curve
    # 초기 로봇이 바라보는 각도
    theta0 = np.pi / 2

    x_ref, y_ref = path
    t = np.arange(0, params.t_length, 0.01)

    # 로봇의 offset을 반영하여 가야 할 초기 좌표를 변형한다.
    # 로봇이 도달해야 하는 좌표 
    # (world frame을 갖는다. x_ref, y_ref에서 offset 뺀값)
    z0 = ptP_to_ptC(x_ref[0],y_ref[0], theta0, params)
    z0 = np.hstack([z0, theta0])
    # theta 추가 - 이건 ㅇㄹ 
    z = z0
    # x, y, theta at the start for point C
    # z0에는 offset 적용되어 있다.
    # z0 = np.array([x0, y0, theta0]); 
    # z0 = np.array(z0)

    # 도달해야 할 point들의 arr 준비
    # p에는 offset 적용이 되어 있지 않음
    traj = np.array([x_ref[0], y_ref[0]])
    # error 값들이 쌓이게 될 arr
    e = np.array([0,0])
    # control signals 
    v = [];
    omega = [];

    for i in range(0,len(t)-1):
    # for i in range(0, 2):
        # 1. get x_c, y_c position
        x_c, y_c, theta = z0;

        # 2. get x_p, y_p from x_c,y_c
        # z0에는 offset 적용되어 있다.
        # 결국 다시 world frame으로 바꾸는 것임 
        # 보통 p0는 센서가 담당한다.
        x_p, y_p = ptC_to_ptP(z0[0], z0[1], z0[2], params)

        #% 3. get error = xe-x_ref and ye-y_ref
        error = [x_ref[i+1]-x_p, y_ref[i+1]-y_p]
        # e = np.vstack([e, error])

        # %4. get u = [v, omega] from the errors
        b = [params.Kp*error[0], params.Kp*error[1]]
        cos = np.cos(theta);
        sin = np.sin(theta);

        px, py = params.px, params.py
        Ainv = np.array([
            [cos-(py/px)*sin, sin+(py/px)*cos],
            [-(1/px)*sin,     (1/px)*cos     ]
        ])
        # u = (v, w)
        u = np.matmul(Ainv, np.transpose(b))
        #u = Ainv*b; #%u = [v omega]
        v.append(u[0])
        omega.append(u[1])

        # 5. now control the car based on u = [v omega]
        z0 = euler_integration([t[i], t[i+1]],z0,[u[0],u[1]])
        z = np.vstack([z, z0])
        p0 = ptC_to_ptP(z0[0],z0[1],z0[2],params);
        traj = np.vstack([traj, p0])

    #interpolation
    t_interp = np.arange(0 , params.t_length , 1/params.fps)
    f_z1 = interpolate.interp1d(t, z[:,0])
    f_z2 = interpolate.interp1d(t, z[:,1])
    f_z3 = interpolate.interp1d(t, z[:,2])
    shape = (len(t_interp),3)
    z_interp = np.zeros(shape)
    z_interp[:,0] = f_z1(t_interp)
    z_interp[:,1] = f_z2(t_interp)
    z_interp[:,2] = f_z3(t_interp)

    f_p1 = interpolate.interp1d(t, traj[:,0])
    f_p2 = interpolate.interp1d(t, traj[:,1])
    shape = (len(t_interp),2)
    p_interp = np.zeros(shape)
    p_interp[:,0] = f_p1(t_interp)
    p_interp[:,1] = f_p2(t_interp)

    animate(t_interp,z_interp,p_interp,params)


if __name__=="__main__":
    params = parameters()
    # "astroid" or "circle"
    path  = generate_path(params, path_type="astroid", show_path=False)
    motion_simulation(params, path)

    pass