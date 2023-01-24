from matplotlib import pyplot as plt
import math
import dd_helper
import numpy as np

class parameters:
    def __init__(self):
        # 목적지에 도달해야 하는 offset
        # 목적지와 딱 맞게 가고 싶다면 px=0.0이어야 하는건가? => yes
        self.px = 0.01
        self.py = 0
        self.Kp = 100;
        # 로봇 반지름
        self.R = 0.1
        self.pause = 0.1
        self.fps = 5
        self.t_length = 10

def animate(params, t_interp, z, p, plot_items):

    # fig2, _ = plt.subplots()

    R = params.R;
    phi = np.arange(0,2*np.pi,0.25)
    x_circle = R*np.cos(phi)
    y_circle = R*np.sin(phi)

    for i in range(0,len(t_interp)):
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
        # shape, = plt.plot(p[0:i,0],p[0:i,1],color='red');
        shape, = plt.plot(z[0:i,0], z[0:i,1], color='red', marker="o", markersize=0.5);

        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.gca().set_aspect('equal')
        plt.pause(params.pause)
        line.remove()
        robot.remove()
        shape.remove()

    fig3, (ax3, ax4, ax5) = plt.subplots(nrows=3, ncols=1)
    e, v, omega = plot_items
    t = np.arange(0, params.t_length, 0.01)
    ax3.set_title("Error (x_ref - x_p)")
    ax3.plot(t, e[:, 0], color="green", label='X err')
    ax3.plot(t, e[:, 1], color="orange", label='Y err')
    ax3.legend(loc="upper right")

    ax4.plot(t, v, color="blue")
    ax4.set_title("Control Signal - V")

    ax5.plot(t, omega, color="red")
    ax5.set_title("Control Signal - W")

    plt.show()

def generate_path(params, path_type="astroid", show_path=False):
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
        fig1, ax1 = plt.subplots()
        ax1.plot(x_ref,y_ref)
        ax1.set_title("Object Path")

        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.gca().set_aspect('equal')

    return x_ref, y_ref

def motion_simulation(params, path):

    x_ref, y_ref = path
    t = np.arange(0, params.t_length, 0.01)

    # initial state
    theta0 = np.pi / 2
    robot_state = x_ref[0], y_ref[0], theta0
    # 로봇이 도달해야 하는 좌표, world frame을 갖는다. x_ref, y_ref에서 offset 뺀값
    z0 = dd_helper.ptP_to_ptC(robot_state, params)
    z0 = np.hstack([z0, theta0])
    z = z0

    # 로봇이 이동하면서 도달한 위치를 저장할 것임
    traj = np.array([x_ref[0], y_ref[0]])

    # plot items - position error & control signals 
    e = np.array([0,0])
    v = [0.0];
    omega = [0.0];

    for i in range(0,len(t)-1):
        # 1. get x_c, y_c position
        x_c, y_c, theta = z0

        # 2. get x_p, y_p from x_c,y_c z0에는 offset 적용되어 있다.
        # 결국 다시 world frame으로 바꾸는 것임 (보통 센서가 담당한다.)
        x_p, y_p = dd_helper.ptC_to_ptP(z0, params)
        # 지금은 아래와 완전 동일함
        # x_p, y_p = x_ref[i], y_ref[i]

        # 3. get error
        error = [x_ref[i+1] - x_p, y_ref[i+1] - y_p]
        e = np.vstack([e, error])

        # 4. get u = [v, omega] from the errors
        # b = [ 10.0 + params.Kp * error[0], 10.0 + params.Kp * error[1]]
        b = [ params.Kp * error[0], params.Kp * error[1]]
        px, py = params.px, params.py

        cos = np.cos(theta)
        sin = np.sin(theta)
        Ainv = np.array([
            [cos-(py/px)*sin, sin+(py/px)*cos],
            [-(1/px)*sin,     (1/px)*cos     ]
        ])
        u = np.matmul(Ainv, np.transpose(b))

        v.append(u[0])
        omega.append(u[1])

        # 5. now control the car based on u = [v omega]
        # offset 적용된 z0로 euler_integration 계산
        z0 = dd_helper.euler_integration([t[i], t[i+1]],z0,[u[0],u[1]])
        z = np.vstack([z, z0])
        # actually useless - It'll be always optimum value
        p0 = dd_helper.ptC_to_ptP(z0, params);
        traj = np.vstack([traj, p0])

    return z, traj, (e, v, omega)

if __name__=="__main__":
    params = parameters()
    # "astroid" or "circle"
    path  = generate_path(params, path_type="astroid", show_path=True)
    
    try:
        # pre calculate motion states
        robot_endpoint, traj, plot_data = motion_simulation(params, path)
    except Exception as e:
        print(e)
    finally:
        # interpolation for animaltion
        t_interp, z_interp, p_interp = dd_helper.interpolation(params, robot_endpoint, traj)
        # draw motion
        animate(params, t_interp,z_interp,p_interp, plot_data)
        print("Everything done!")
