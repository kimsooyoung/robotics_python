
import numpy as np

from scipy import interpolate
from scipy.integrate import odeint

from one_step import one_step
from visualize import animate, plot
from utils import find_fixed_points

class Parameters:
    
    def __init__(self):
        
        self.dof = 14
        
        # leg lengths, body width
        self.l0 = 1; self.l1 = 0.5; self.l2 = 0.5
        self.w = 0.1
        
        # mass and inertia
        self.mb = 70; self.mt = 10; self.mc = 5
        self.Ibx = 5; self.Iby = 3; self.Ibz = 2
        self.Itx = 1; self.Ity = 0.3; self.Itz = 2
        self.Icx = 0.5; self.Icy = 0.15; self.Icz = 1
        
        self.g = 9.8

        # total weight and height
        self.M = self.mb + 2*self.mt + 2*self.mc
        self.L = self.l1 + self.l2
        
        # misc other params
        self.stepAngle = 0.375 # #step length
        self.Impulse = 0.18 * self.M * np.sqrt(self.g * self.L) #push-off impulse
        self.kneeAngle = -1 #Knee bending to avoid scuffing
        
        self.Kp = 100 #gain for partial feedback linearization
        self.Kd = np.sqrt(self.Kp) 
        
        self.stance_foot_init = 'right'
        self.stance_foot = self.stance_foot_init
        
        self.fps = 10
        self.pause = 0.01

if __name__=="__main__":

    params = Parameters()
    
    phi, theta, psi, psid, thetad, phid = 0, 0, 0, 0, 0.5, 0
    psi_lh, phi_lh, theta_lh, theta_lk = 0, 0, 0, 0
    psi_rh, phi_rh, theta_rh, theta_rk = 0, 0, 0, 0
    psi_lhd, phi_lhd, theta_lhd, theta_lkd = 0, 0, 0, 0
    psi_rhd, phi_rhd, theta_rhd, theta_rkd = 0, 0, 0, 0
    
    if params.stance_foot_init == 'right':
        theta_rk = params.kneeAngle
    elif params.stance_foot_init == 'left':
        theta_lk = params.kneeAngle
    
    # 6 + 12 + 4 = 22
    z0 = np.array([phi, phid, theta, thetad, psi, psid, \
                     phi_lh, phi_lhd, theta_lh, theta_lhd, psi_lh, psi_lhd, theta_lk, theta_lkd, \
                         phi_rh, phi_rhd, theta_rh, theta_rhd, psi_rh, psi_rhd, theta_rk, theta_rkd])
    
    # # 1) find fixed points and check stability
    # z_star = find_fixed_points(z0, params)
    # print(f"z_star: {z_star}")
    # pass
    
    # Fixed point z_star : 
    # [-3.62077919e-10 -1.26645439e+07 -8.15710858e-10  1.78608084e+07
    # 2.89783776e-09 -1.36538168e+07 -1.27493502e-10 -1.22551512e+07
    # 8.17699271e-09  4.04836472e+07 -5.74888291e-11  4.19244188e+07
    # -2.21641530e-16 -6.29194119e+07  8.85956111e-09  2.71104967e+07
    # -4.30591757e-08  2.39587331e+06  4.34842142e-09 -7.49797939e+04
    # -9.99999980e-01  2.14851269e+07]

    z_star = np.array([ 
        0.000000000000000,
        0.000000000000000,
        -0.000000000000000,
        0.000000000000001,
        -0.000000000000006,
        -0.000000000000003,
        0.000000000000006,
        0.000000000000043,
        -0.000000000000002,
        0.000000000000070,
        -0.000000000000013,
        -0.000000000000057,
        -0.999999999999981,
        0.000000000000061,
        -0.029247773468329,
        0.054577886084082,
        -0.001551040824746,
        -1.029460778006675,
        0.054690088280553,
        0.559306810759302,
        -0.000000000000001,
        -0.000000000000031
    ])
    
    # 2) forward simulation
    steps = 4
    params.stance_foot = params.stance_foot_init
    Z, t, P_LA_all, P_RA_all, Torque = one_step(z_star, params, steps)
    print('----- start state --------- end state ----')
    
    # TODO: collision dbg
    print(np.hstack((z_star.reshape(-1, 1), Z[-1, 6:].reshape(-1, 1))))
    print(Z.shape)
    print(Torque.shape)

    # print(f"Z[:,0]: {Z[:,0]}")

    # TODO: plot torque
    # 3) plotting and animation
    view = (-55, 30)
    animate(t, Z, params, view)
    # plot(t, Z, Torque, params)
    
    # [Z,t,P_LA_all,P_RA_all,Torque] = onestep(zstar,parms,steps);

    # disp('----- start state --------- end state ----');
    # disp([zstar' Z(end,7:end)']);

    # # 3) plotting and animation
    
    
    # # fps = 50;
    # # figure(1)
    # # title('animation');
    # # % view([0 0]);
    # # % view([71 17]);
    # # view([60 54]);
    # # animate(t,Z,parms,fps,view)
