from matplotlib import pyplot as plt
import numpy as np

from scipy import interpolate
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from copy import deepcopy

def cos(x):
    return np.cos(x)

def sin(x):
    return np.sin(x)

class Parameters:
    def __init__(self):
        # m, M : leg mass, body mass
        # I : body moment of inertia
        # g : gravity
        # c, l : leg length, body length
        # gam : slope angle
        # P : push force from walker foot
        # pause, fps : var for animation
        
        self.M = 1.0
        self.m = 0.5
        self.I = 0.02
        self.l = 1.0
        self.c = 0.5
        self.g = 1.0
        self.gam = 0*0.01

        self.t_opt = [0.,         0.61236711, 1.22473421, 1.83710132, 2.44946843]
        self.u_opt = [ 1.30154083e-06, 1.02086218e-06, -7.97604653e-07, -2.76868629e-06, -1.27223475e-06]
        
        self.P = 0.1
        
        self.pause = 0.01
        self.fps = 10

# output이 0이면 충돌이 일어났다는 뜻
def collision(t, z, M, m, I, l, c, g, gam, t_opt, u_opt):

    output = 1
    theta1, omega1, theta2, omega2 = z

    if (theta1 > -0.05):
        output = 1
    else:
        output = 2 * theta1 + theta2

    return output

# torque powered 
def single_stance(t, z, M, m, I, l, c, g, gam, t_opt, u_opt):
    
    theta1, omega1, theta2, omega2 = z
    
    f = interpolate.interp1d(t_opt, u_opt)
    Th = f(t)
    
    A = np.zeros((2,2))
    b = np.zeros((2,1))

    A[0,0] = 2.0*I + M*l**2 + m*(c - l)**2 + m*(c**2 - 2*c*l*cos(theta2) + l**2)
    A[0,1] = 1.0*I + c*m*(c - l*cos(theta2))
    A[1,0] = 1.0*I + c*m*(c - l*cos(theta2))
    A[1,1] = 1.0*I + c**2*m

    b[0] = -M*g*l*sin(gam - theta1) + c*g*m*sin(gam - theta1) - c*g*m*sin(-gam + theta1 + theta2) - 2*c*l*m*omega1*omega2*sin(theta2) - c*l*m*omega2**2*sin(theta2) - 2*g*l*m*sin(gam - theta1)
    b[1] = 1.0*Th - 1.0*c*g*m*sin(-gam + theta1 + theta2) + 1.0*c*l*m*omega1**2*sin(theta2)

    alpha1, alpha2 = np.linalg.inv(A).dot(b)
    
    return [ omega1, alpha1, omega2, alpha2 ]

def single_stance_ode_int(z, t, M, m, I, l, c, g, gam, t1, t2, u1, u2):
    
    theta1, omega1, theta2, omega2 = z
    
    Th = u1 + (u2-u1)/(t2-t1)*(t-t1)
    
    A = np.zeros((2,2))
    b = np.zeros((2,1))

    A[0,0] = 2.0*I + M*l**2 + m*(c - l)**2 + m*(c**2 - 2*c*l*cos(theta2) + l**2)
    A[0,1] = 1.0*I + c*m*(c - l*cos(theta2))
    A[1,0] = 1.0*I + c*m*(c - l*cos(theta2))
    A[1,1] = 1.0*I + c**2*m

    b[0] = -M*g*l*sin(gam - theta1) + c*g*m*sin(gam - theta1) - c*g*m*sin(-gam + theta1 + theta2) - 2*c*l*m*omega1*omega2*sin(theta2) - c*l*m*omega2**2*sin(theta2) - 2*g*l*m*sin(gam - theta1)
    b[1] = 1.0*Th - 1.0*c*g*m*sin(-gam + theta1 + theta2) + 1.0*c*l*m*omega1**2*sin(theta2)

    alpha1, alpha2 = np.linalg.inv(A).dot(b)
    
    return [ omega1, alpha1, omega2, alpha2 ]

def footstrike(t_minus, z_minus, params):

    theta1_n, omega1_n, theta2_n, omega2_n = z_minus
    
    M = params.M
    m = params.m
    I = params.I
    l = params.l
    c = params.c
    g = params.g
    gam = params.gam
    P = params.P

    theta1_plus = theta1_n + theta2_n
    theta2_plus = -theta2_n

    J_n_sw = np.zeros((2,4))
    J_n_st = np.zeros((2,4))
    P_st = np.zeros((2,1))
    
    A_n_hs = np.zeros((4,4))
    b_hs = np.zeros((6,1))

    J11 =  1
    J12 =  0
    J13 =  l*(-cos(theta1_n) + cos(theta1_n + theta2_n))
    J14 =  l*cos(theta1_n + theta2_n)
    J21 =  0
    J22 =  1
    J23 =  l*(-sin(theta1_n) + sin(theta1_n + theta2_n))
    J24 =  l*sin(theta1_n + theta2_n)
    J_n_sw = np.array([[J11, J12, J13, J14], [J21,J22,J23,J24]])
    
    J11 =  1
    J12 =  0
    J13 =  0
    J14 =  0
    J21 =  0
    J22 =  1
    J23 =  0
    J24 =  0 
    J_n_st = np.array([[J11, J12, J13, J14], [J21,J22,J23,J24]]) 

    # strike시 고정된 발끝에서 impluse 추가 
    P_st[0] = -P*sin(theta1_n)
    P_st[1] = P*cos(theta1_n)

    A11 =  1.0*M + 2.0*m
    A12 =  0
    A13 =  -1.0*M*l*cos(theta1_n) + m*(c - l)*cos(theta1_n) + 1.0*m*(c*cos(theta1_n + theta2_n) - l*cos(theta1_n))
    A14 =  1.0*c*m*cos(theta1_n + theta2_n)
    A21 =  0
    A22 =  1.0*M + 2.0*m
    A23 =  -1.0*M*l*sin(theta1_n) + m*(c - l)*sin(theta1_n) + m*(c*sin(theta1_n + theta2_n) - l*sin(theta1_n))
    A24 =  1.0*c*m*sin(theta1_n + theta2_n)
    A31 =  -1.0*M*l*cos(theta1_n) + m*(c - l)*cos(theta1_n) + 1.0*m*(c*cos(theta1_n + theta2_n) - l*cos(theta1_n))
    A32 =  -1.0*M*l*sin(theta1_n) + m*(c - l)*sin(theta1_n) + m*(c*sin(theta1_n + theta2_n) - l*sin(theta1_n))
    A33 =  2.0*I + M*l**2 + m*(c - l)**2 + m*(c**2 - 2*c*l*cos(theta2_n) + l**2)
    A34 =  1.0*I + c*m*(c - l*cos(theta2_n))
    A41 =  1.0*c*m*cos(theta1_n + theta2_n)
    A42 =  1.0*c*m*sin(theta1_n + theta2_n)
    A43 =  1.0*I + c*m*(c - l*cos(theta2_n))
    A44 =  1.0*I + c**2*m
    A_n_hs = np.array([[A11, A12, A13, A14], [A21, A22, A23, A24], [A31, A32, A33, A34], [A41, A42, A43, A44]])

    A_hs = np.block([
        [A_n_hs, -np.transpose(J_n_sw) ], 
        [J_n_sw, np.zeros((2,2))] 
    ])
    
    X_n_hs = np.zeros((4,1))
    X_n_hs[0,0] = 0; X_n_hs[1,0] = 0; 
    X_n_hs[2,0] = omega1_n; X_n_hs[3,0] = omega2_n
    
    b_hs = np.block([
        [ A_n_hs@X_n_hs + J_n_st.T@P_st ],
        [ np.zeros((2,1)) ]
    ])

    # x_hs => [vx(+), vy(+), omega1(+), omega2(+) ]
    x_hs = np.linalg.inv(A_hs).dot(b_hs)

    omega1_plus = x_hs[2,0] + x_hs[3,0]
    omega2_plus = -x_hs[3,0]
    
    # print(omega1_plus, omega2_plus)
    
    return [theta1_plus, omega1_plus, theta2_plus, omega2_plus]

def one_step(z0, t0, params):

    t_start = t0
    t_end   = t_start + params.t_opt[-1]
    print(params.t_opt[-1])
    t = np.linspace(t_start, t_end, 100)

    sol = solve_ivp(
        single_stance, [t_start, t_end], z0, method='RK45', t_eval=t,
        dense_output=True, atol = 1e-13, rtol = 1e-13, 
        args=(
            params.M,params.m,params.I,
            params.l,params.c,params.g,params.gam,
            params.t_opt, params.u_opt
        )
    )

    t = sol.t
    # m : 4 / n : 1001
    m, n = np.shape(sol.y)
    z = np.zeros((n, m))
    z = sol.y.T

    # for gait optimization 
    print(f"z bf strike : {z[-1]}")
    
    ######### 여기까지 single stance #########
    z_minus = z[-1]
    z_plus = footstrike(0, z_minus, params)
    z[-1] = z_plus

    return z, t

def n_steps(z0, t0, step_size, params):

    # xh_start, yh_start : hip position
    xh_start, yh_start = 0, params.l * cos(z0[0])

    t = np.array([t0])
    z = np.zeros((1, 6))
    z[0] = np.append(z0, np.array([xh_start, yh_start]))

    for i in range(step_size):
        z_temp, t_temp = one_step(z0, t0, params)
        
        zz_temp = np.zeros((len(t_temp), 6))

        # append xh, yh - hip position
        for j in range(len(t_temp)):
            xh = xh_start + params.l * sin(z_temp[0,0]) - params.l * sin(z_temp[j,0])
            yh = params.l * cos(z_temp[j,0])
            # z_temp[j,:] = np.append(z_temp[j,:], np.array([xh, yh]))
            # 이렇게 하고 싶지만 numpy array는 초기에 크기가 정해지면 append가 안된다.
            # ValueError: could not broadcast input array from shape (6,) into shape (4,)
            zz_temp[j,:] = np.append(z_temp[j,:], np.array([xh, yh]))

        z = np.concatenate((z, zz_temp), axis=0)
        t = np.concatenate((t, t_temp), axis=0)

        theta1, omega1, theta2, omega2 = z_temp[-1,0:4]
        z0 = np.array([theta1, omega1, theta2, omega2])
        t0 = t_temp[-1]

        # one step에서 zz_temp[-1] 스위칭이 일어나기 때문에 [-2] 사용
        xh_start = zz_temp[-2,4]

        # debug : only one step without strike
        # break
        
    return z, t

def animate(t,z,parms):
    #interpolation
    data_pts = 1/parms.fps
    t_interp = np.arange(t[0],t[len(t)-1],data_pts)

    [m,n] = np.shape(z)
    z_interp = np.zeros((len(t_interp),n))

    for i in range(0,n):
        f = interpolate.interp1d(t, z[:,i])
        z_interp[:,i] = f(t_interp)

    l = parms.l
    c = parms.c

    min_xh = min(z[:,4]) 
    max_xh = max(z[:,4])

    dist_travelled = max_xh - min_xh;
    camera_rate = dist_travelled/len(t_interp);

    window_xmin = -1*l; window_xmax = 1*l;
    window_ymin = -0.1; window_ymax = 1.1*l;

    R1 = np.array([min_xh-l,0])
    R2 = np.array([max_xh+l,0])

    # 바닥은 처음에 다 그려버린다.
    ramp, = plt.plot([R1[0], R2[0]],[R1[1], R2[1]],linewidth=5, color='black')
    
    # plot body
    for i in range(0,len(t_interp)):
        theta1 = z_interp[i,0];
        theta2 = z_interp[i,2];
        xh = z_interp[i,4];
        yh = z_interp[i,5];

        H = np.array([xh, yh])
        C1 = np.array([xh+l*sin(theta1), yh-l*cos(theta1)])
        G1 = np.array([xh+c*sin(theta1), yh-c*cos(theta1)])
        C2 = np.array([xh+l*sin(theta1+theta2),yh-l*cos(theta1+theta2)])
        G2 = np.array([xh+c*sin(theta1+theta2),yh-c*cos(theta1+theta2)])

        leg1, = plt.plot([H[0], C1[0]],[H[1], C1[1]],linewidth=5, color='red')
        leg2, = plt.plot([H[0], C2[0]],[H[1], C2[1]],linewidth=5, color='red')
        com1, = plt.plot(G1[0],G1[1],color='black',marker='o',markersize=5)
        com2, = plt.plot(G2[0],G2[1],color='black',marker='o',markersize=5)
        hip, = plt.plot(H[0],H[1],color='black',marker='o',markersize=10)

        # camera_rate 만큼 화면을 오른쪽으로 이동시킨다.
        window_xmin = window_xmin + camera_rate;
        window_xmax = window_xmax + camera_rate;
        plt.xlim(window_xmin,window_xmax)
        plt.ylim(window_ymin,window_ymax)
        plt.gca().set_aspect('equal')

        plt.pause(parms.pause)
        hip.remove()
        leg1.remove()
        leg2.remove()
        com1.remove()
        com2.remove()

    plt.close()

def fixedpt(z0, params):

    z, t = one_step(z0, 0, params)

    return z[-1,0]-z0[0], z[-1,1]-z0[1], z[-1,2]-z0[2], z[-1,3]-z0[3]

def partial_jacobian(z, params):

    m = len(z)
    J = np.zeros((m, m))

    epsilon = 1e-5

    for i in range(m):
        # LIST IS IMMUATABLE
        z_minus = deepcopy(z)
        z_plus  = deepcopy(z)

        z_minus[i] = z[i] - epsilon
        z_plus[i]  = z[i] + epsilon

        z_minus_result, _ = one_step(z_minus, 0, params)
        z_plus_result, _  = one_step(z_plus, 0, params)

        for j in range(m):
            J[i,j] = (z_plus_result[-1,j] - z_minus_result[-1,j]) / (2 * epsilon)

    return J

def plot(t, z):
    plt.figure(1)
    plt.subplot(2,1,1)

    plt.plot(t,z[:,0],'r--', label=r'$\theta_1$')
    plt.plot(t,z[:,2],'b', label=r'$\theta_2$')
    plt.ylabel("position")
    plt.legend(loc=(1.0, 1.0), ncol=1, fontsize=7)
    
    plt.subplot(2,1,2)
    plt.plot(t,z[:,1],'r--', label=r'$\dot{\theta_1}$')
    plt.plot(t,z[:,3],'b', label=r'$\dot{\theta_2}$')
    plt.ylabel("velocity")
    plt.legend(loc=(1.0, 1.0), ncol=1, fontsize=7)
    plt.xlabel('time')

    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(t,z[:,4],'b')
    plt.ylabel('xh')
    
    plt.subplot(2,1,2)
    plt.plot(t,z[:,5],'b')
    plt.ylabel('yh')
    plt.xlabel('time')

    plt.figure(3)
    plt.subplot(2,1,1)
    plt.plot(z[:,0],z[:,1],'r', linewidth=3)
    plt.plot(z[0,0],z[0,1],'ko', markersize=10, markerfacecolor='k')
    plt.ylabel(r'$\dot{\theta_1}$')
    plt.xlabel(r'$\theta_1$')
    
    plt.subplot(2,1,2)
    plt.plot(z[:,0]+z[:,2], z[:,1]+z[:,3],'b', linewidth=2)
    plt.plot(z[0,0]+z[0,2],z[0,1]+z[0,3],'ko', markersize=10, markerfacecolor='k')
    plt.ylabel('yh')
    plt.ylabel(r'$\dot{\theta_1} + \dot{\theta_2}$')
    plt.xlabel(r'$\theta_1 + \theta_2$')

    # # trajectory log
    # plt.figure(4)
    # plt.subplot(2,1,1)
    # plt.plot(t, z_ref[:,0],'k-', linewidth=2, label="reference")
    # plt.plot(t, z[0,2], 'r', label="actual")
    # plt.ylabel(r'$\theta_2$')
    
    # plt.subplot(2,1,2)
    # plt.plot(t, z_ref[:,1],'k-', linewidth=2, label="reference")
    # plt.plot(t, z[0,3], 'r', label="actual")
    # plt.ylabel(r'$ \dot{\theta_2} $')
    
    # plt.show()
    plt.show(block=False)
    plt.pause(3)
    plt.close()

if __name__=="__main__":
    
    params = Parameters()

    #####################################
    ######## initial state ##############
    #####################################

    is_opt = True
    
    if is_opt:
        # opt value from slsqp
        # theta1, omega1, theta2, omega2 = 0.17968459261613756, -0.2710492924103855, -0.3593691852322751, 0.026444172032221862
        # params.t_opt = [0.         ,0.59653683, 1.19307366 ,1.78961048 ,2.38614731]
        # params.u_opt = [-5.58706476e-05, -4.18215580e-04, -6.02068358e-04, -4.05763968e-04, -1.05010935e-04]
        
        # ex b
        theta1, omega1, theta2, omega2 = 0.25268025514416304, -1.1561444475441687, -0.5053605102883261, 0.11848173806190909
        params.t_opt = [0.,    0.125, 0.25,  0.375, 0.5  ]
        params.u_opt = [0.89897521, 1.47418862, 1.0134691, 0.56638579, 0.13636414]
        params.P = 0.3139217651577935
    else:
        theta1, omega1, theta2, omega2 = 0.2, -0.25, -0.4, 0.2
    
    t0 = 0
    
    if is_opt:
        step_size = 1
    else:
        step_size = 5

    z0 = np.array([theta1, omega1, theta2, omega2])

    ##########################################
    ### 실패하지 않는 초기 조건을 찾아보자. ####
    ##########################################
    if is_opt:
        z_star = z0
    else:
        z_star = fsolve(fixedpt, z0, params)
        print(f"z_star : {z_star}")
        # 해당 초기 조건에 대한 stability를 확인해보자.
        # Jacobian의 determinant를 통해 구해야 하는데, 
        # Jacobian을 대수적으로 구할 수 없으므로 수치적으로 구해볼 것이다.
        J_star = partial_jacobian(z_star, params)
        eig_val, eig_vec = np.linalg.eig(J_star)
        print(f"eigVal {eig_val}")
        print(f"eigVec {eig_vec}")
        print(f"max(abs(eigVal)) : {max(np.abs(eig_val))}")
        
    z, t = n_steps(z_star, t0, step_size, params)
    
    if is_opt:
        print(f"\n====== Check if those two vals are same ======")
        print(f"z_star : {z_star}")
        print(f"z_end : {z[-1][:-2]}")
        
    # animate(t, z, params)
    plot(t, z)
