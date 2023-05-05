import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# TODO:
# 1. flight eom
#    - contact event
#    - apex event
# 2. stance eom
#    - release event
#
# 3. one bounce
#
# 4. animation

class Params:
    def __init__(self):
        self.g = 10.0
        self.ground = 0.0
        self.l = 1
        self.m = 1
        
        # sprint stiffness
        # self.k = 200
        self.k = 100
        # fixed angle
        self.theta = 10 * (np.pi / 180)
        
        self.pause = 0.1
        self.fps = 10
        
        # new params for raibert hopper
        self.T = np.pi * np.sqrt(self.m/self.k)
        
        # TODO: Adjust Kp 
        self.Kp = 0.1
        # self.vdes = [0] * 3
        # self.vdes = [0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        self.vdes = [0.0, 0.2]

def flight(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    
    return [x_dot, 0, y_dot, -g]

def contact(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    # contact event
    return y - l0 * np.cos(theta)
# contact.direction = -1
# contact.terminal = True

def apex(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    return y_dot - 0
# apex.direction = 0
# apex.terminal = True

def stance(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    
    l = np.sqrt(x**2 + y**2)
    F_spring = k * (l0 - l)
    # F_spring = k * (l - l0)
    Fx_spring = F_spring * (x / l)
    Fy_spring = F_spring * (y / l)
    Fy_gravity = m*g
    
    x_dd = (1/m)*(Fx_spring)
    y_dd = (1/m)*(-Fy_gravity + Fy_spring)
    
    return [x_dot, x_dd, y_dot, y_dd]

def release(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    l = np.sqrt(x**2 + y**2)
    
    return l - l0
# release.direction = +1
# release.terminal = True

def onestep(z0, t0, params):
    
    dt = 5
    x, x_d, y, y_d = z0
    m, g, k = params.m, params.g, params.k
    l0, theta = params.l, params.theta
    
    t_output = np.zeros(1)
    t_output[0] = t0
    # z_output = [x, x_dot, y, y_dot, x_foot, y_foot]
    z_output = np.zeros((1,6))
    z_output[0] = [*z0, x+l0*np.sin(theta), y-l0*np.cos(theta)]

    #####################################
    ###         contact phase         ###
    #####################################
    contact.direction = -1
    contact.terminal = True
    
    ts = np.linspace(t0, t0+dt, dt*1000)

    # def contact(t, z, l0, theta):
    contact_sol = solve_ivp(
        flight, [t0, t0+dt], z0, method='RK45', t_eval=np.linspace(t0, t0+dt, 1001),
        dense_output=True, events=contact, atol = 1e-13, rtol = 1e-13, 
        args=(m, g, l0, k, theta)
    )
    
    t_contact = contact_sol.t
    m, n = contact_sol.y.shape
    z_contact = contact_sol.y.T
    
    # calculate foot position for animation
    x_foot = z_contact[:,0] + l0*np.sin(theta)
    y_foot = z_contact[:,2] - l0*np.cos(theta)

    # append foot position into z vector
    z_contact_output = np.concatenate((z_contact, x_foot.reshape(-1,1), y_foot.reshape(-1,1)), axis=1)
    
    # add to output
    t_output = np.concatenate((t_output, t_contact[1:]))
    z_output = np.concatenate((z_output, z_contact_output[1:]))

    #####################################
    ## adjust new state for next phase ##
    #####################################
    t0, z0 = t_contact[-1], z_contact[-1]
    # print(f"z_contact: {z0}")
    # save the x position for future
    x_com = z0[0]
    # relative distance wrt contact point because of 
    # non-holonomic nature of the system
    z0[0] = -l0*np.sin(theta)
    # print(f"z_contact2: {z0}")
    x_foot_gnd = x_com + l0*np.sin(theta)
    y_foot_gnd = params.ground

    #####################################
    ###          stance phase         ###
    #####################################
    release.direction = 1
    release.terminal = True

    # TODO: robust stance
    ts = np.linspace(t0, t0+dt, dt*2000)
    # def stance(t, z, m, g, l0, k, theta)
    release_sol = solve_ivp(
        stance, [t0, t0+dt], z0, t_eval=np.linspace(t0, t0+dt, 1001),
        dense_output=True, events=release, atol = 1e-13, rtol = 1e-13, 
        args=(m, g, l0, k, theta)
    )

    t_release = release_sol.t
    m, n = release_sol.y.shape
    z_release = release_sol.y.T
    # print(f"z_release1 : {z_release[-1]}")
    z_release[:,0] = z_release[:,0] + x_com + l0*np.sin(theta)

    # append foot position for animation
    x_foot = x_foot_gnd * np.ones((n,1))
    y_foot = y_foot_gnd * np.ones((n,1))
    z_release_output = np.concatenate((z_release, x_foot, y_foot), axis=1)
    
    # add to output
    t_output = np.concatenate((t_output, t_release[1:]))
    z_output = np.concatenate((z_output, z_release_output[1:]))

    #####################################
    ## adjust new state for next phase ##
    #####################################
    t0, z0 = t_release[-1], z_release[-1]
    # print(f"z_release : {z0}")

    #####################################
    ###           apex  phase         ###
    #####################################
    apex.direction = -1
    apex.terminal = True

    ts = np.linspace(t0, t0+dt, dt*1000)
    # def stance(t, z, m, g, l0, k, theta)
    apex_sol = solve_ivp(
        flight, [t0, t0+dt], z0, method='RK45', t_eval=np.linspace(t0, t0+dt, 1001),
        dense_output=True, events=apex, atol = 1e-13, rtol = 1e-13, 
        args=(m, g, l0, k, theta)
    )

    t_apex = apex_sol.t
    m, n = apex_sol.y.shape
    z_apex = apex_sol.y.T

    # calculate foot position for animation
    x_foot = z_apex[:,0] + l0*np.sin(theta)
    y_foot = z_apex[:,2] - l0*np.cos(theta)
    z_apex_output = np.concatenate((z_apex, x_foot.reshape(-1,1), y_foot.reshape(-1,1)), axis=1)

    # add to output
    t_output = np.concatenate((t_output, t_apex[1:]))
    z_output = np.concatenate((z_output, z_apex_output[1:]))
    
    # print(f"z_apex : {z_output[-1]}")

    return z_output, t_output

def n_step(zstar, params):
    
    # x0 = 0; x0dot = z0(1);  
    # y0 = z0(2); y0dot = 0;
    
    z0 = zstar
    t0 = 0

    z = np.zeros((1,6))
    t = np.zeros(1)

    steps = len(params.vdes)
    v_apex = []
    
    i = 0    
    for i in range(steps):
        
        vdes = params.vdes[i]
        x0dot = z0[1]
        v_apex.append(x0dot)
        # print(v_apex)
        print(f"x0dot: {x0dot}")
        
        params.theta = np.arcsin((x0dot * params.T)/(2*params.l)) + \
            params.Kp*(x0dot - vdes)
        print(f"theta : {params.theta}")
        
        if i == 0:
            z, t = onestep(z0, t0, params)
            print(f'apex horizontal velocity: {vdes}')
        else:
            z_step, t_step = onestep(z0, t0, params)
            z = np.concatenate((z, z_step[1:]))
            t = np.concatenate((t, t_step[1:]))

        z0 = z[-1][:-2]
        t0 = t[-1]
    
    return z, t, v_apex

def animate(z, t, parms):
    #interpolation
    data_pts = 1/parms.fps
    t_interp = np.arange(t[0], t[len(t)-1], data_pts)
    m, n = np.shape(z)
    shape = (len(t_interp),n)
    z_interp = np.zeros(shape)

    for i in range(0, n):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    l = parms.l

    min_xh = min(z[:,0]); max_xh = max(z[:,0]);
    # print(f"min_xh: {min_xh}, max_xh: {max_xh}")
    dist_travelled = max_xh - min_xh;
    camera_rate = dist_travelled/len(t_interp);

    window_xmin = -1*l; window_xmax = 1*l;
    window_ymin = -0.1; window_ymax = 1.9*l;

    #plot
    for i in range(0,len(t_interp)):

        x, y = z_interp[i,0], z_interp[i,2]
        x_foot, y_foot = z_interp[i,4], z_interp[i,5]

        leg, = plt.plot([x, x_foot],[y, y_foot],linewidth=2, color='black')
        hip, = plt.plot(x, y, color='red', marker='o', markersize=10)

        window_xmin = window_xmin + camera_rate;
        window_xmax = window_xmax + camera_rate;
        plt.xlim(window_xmin,window_xmax)
        plt.ylim(window_ymin,window_ymax)
        plt.gca().set_aspect('equal')

        plt.pause(parms.pause)
        hip.remove()
        leg.remove()

    plt.close()

def plot(z, t, params, v_apex):
    plt.figure(1)
    
    plt.subplot(2,1,1)
    plt.plot(t,z[:,0],'r--')
    plt.plot(t,z[:,2],'b')
    plt.ylabel('Position (m)')
    plt.xlabel('time')
    
    plt.subplot(2,1,2)
    plt.plot(t,z[:,1],'r--')
    plt.plot(t,z[:,3],'b')
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('time')

    plt.figure(2)
    plt.title('Trajectory y vs x')
    plt.plot(z[:,0],z[:,2],'r')
    plt.ylabel('y')
    plt.xlabel('x')

    plt.figure(3)
    plt.step(params.vdes,'k--', linewidth=2, label='Desired')
    plt.step(v_apex,'r', linewidth=2, label='Actual')
    plt.legend()

    plt.show(block=False)
    plt.pause(3)
    plt.close()
    
    # ### 
    # figure(4)
    # stairs(robot.control.vdes,'k--','Linewidth',2); hold on
    # stairs(v_apex,'r','Linewidth',2); 
    # legend('desired','actual');

def partialder(z0, parms):

    pert = 1e-5
    N = len(z0)

    J = np.zeros((N,N))
    z_temp1 = [0]*N
    z_temp2 = [0]*N
    for i in range(0,N):
        for j in range(0,N):
            z_temp1[j] = z0[j]
            z_temp2[j] = z0[j]
        z_temp1[i] = z_temp1[i]+pert
        z_temp2[i] = z_temp2[i]-pert
        [z1,t1] = onestep(z_temp1, 0, parms)
        [z2,t2] = onestep(z_temp2, 0, parms)
        for j in range(0,N):
            J[i,j] = (z1[len(t1)-1,j] - z2[len(t2)-1,j])/(2*pert)

    return J

def fixedpt(z0, parms):
    t0 = 0
    z1, t1 = onestep(z0, t0, parms)
    N = len(t1)-1

    return z1[N,0]-z0[0], z1[N,1]-z0[1],z1[N,2]-z0[2],z1[N,3]-z0[3]

if __name__=="__main__":
    
    params = Params()

    x0_d = params.vdes[0]
    params.theta = np.arcsin((x0_d * params.T)/(2*params.l)) + \
        params.Kp*(x0_d - params.vdes[0])        
    print(f"theta : {params.theta}")

    # x, x_d, y, y_d = 0, 0.34271, 1.1, 0
    x, x_d, y, y_d = 0, x0_d, 1.2, 0
    z0 = np.array([x, x_d, y, y_d])

    # # 실패하지 않는 초기 조건을 찾아보자. Jacobian의 최대 eigenvalue를 통해 판별할 수 있다.
    # # 
    # # max(eig(J)) < 1 => stable
    # # max(eig(J)) = 1 => neutrally stable
    # # max(eig(J)) > 1 => unstable
    # zstar = fsolve(fixedpt, z0, params)
    # print(f"zstar : {zstar}")
    
    # # Jacobian을 구할 수식이 없다. 따라서 수치적으로 구해본다.
    # J = partialder(zstar, params)
    # eigVal, eigVec = np.linalg.eig(J)
    # print(f"eigVal {eigVal}")
    # print(f"eigVec {eigVec}")
    # print(f"abs(eigVal) : {np.abs(eigVal)}")

    z, t, v_apex = n_step(z0, params)
    # animate(z, t, params)
    plot(z, t, params, v_apex)