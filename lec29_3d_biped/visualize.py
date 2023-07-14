import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt
from scipy import interpolate
import numpy as np

# from joint_locations import joint_locations
from cython_dynamics.joint_locations_cython import joint_locations

def animate(t_all, z_all, params, view):
    
    l0, l1, l2 = params.l0, params.l1, params.l2
    w = params.w
    
    m, n = z_all.shape
    print(f"m: {m}, n: {n}")
    n_state = int(n/2)
    
    z_all_plot = np.zeros((m, n_state ))
    for i in range(0, n, 2):
        z_all_plot[:, int(i/2)] = z_all[:, i]
    
    total_frames = round(t_all[-1] * params.fps)
    zz = np.zeros((total_frames, n_state))
    t = np.linspace(0, t_all[-1], total_frames)

    for i in range(n_state):
        f = interpolate.interp1d(t_all, z_all_plot[:, i])
        zz[:, i] = f(t)
    
    # print(f"zz[:,0] : {zz[:,0]}")
    
    mm, nn = zz.shape
    # print(f"mm: {mm}, nn: {nn}")
    
    fig = plt.figure(1)
    
    for i in range(mm):
        
        # For MacOS Users
        # ax = p3.Axes3D(fig)

        # For Windows/Linux Users
        ax = fig.add_subplot(111, projection='3d')
        
        azim, elev = view
        ax.view_init(azim=azim,elev=elev)

        j = 0
        x = zz[i, j]; j += 1
        y = zz[i, j]; j += 1
        z = zz[i, j]; j += 1
        phi = zz[i, j]; j += 1
        theta = zz[i, j]; j += 1
        psi = zz[i, j]; j += 1
        phi_lh = zz[i, j]; j += 1
        theta_lh = zz[i, j]; j += 1
        psi_lh = zz[i, j]; j += 1
        theta_lk = zz[i, j]; j += 1
        phi_rh = zz[i, j]; j += 1
        theta_rh = zz[i, j]; j += 1
        psi_rh = zz[i, j]; j += 1
        theta_rk = zz[i, j]
        
        B,H,LH,LK,LA,RH,RK,RA,b,rt,rc,lt,lc = joint_locations(l0,l1,l2,phi,phi_lh,phi_rh,psi_lh,psi_rh,psi,theta,theta_lh,theta_lk,theta_rh,theta_rk,w,x,y,z);
    
        loc1 = B; loc2 = H;
        k1, = ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], [loc1[2], loc2[2]],linewidth=5, color='r')
        
        loc3 = B; loc4 = LH;
        k2, = ax.plot([loc3[0], loc4[0]], [loc3[1], loc4[1]], [loc3[2], loc4[2]],linewidth=5, color='b')

        loc5 = LH; loc6 = RH;
        k3, = ax.plot([loc5[0], loc6[0]], [loc5[1], loc6[1]], [loc5[2], loc6[2]],linewidth=5, color='b')
    
        loc7 = LH; loc8 = LK;
        k4, = ax.plot([loc7[0], loc8[0]], [loc7[1], loc8[1]], [loc7[2], loc8[2]],linewidth=5, color='c')
        
        loc9 = RH; loc10 = RK;
        k5, = ax.plot([loc9[0], loc10[0]], [loc9[1], loc10[1]], [loc9[2], loc10[2]],linewidth=5, color='c')
        
        loc11 = LK; loc12 = LA;
        k6, = ax.plot([loc11[0], loc12[0]], [loc11[1], loc12[1]], [loc11[2], loc12[2]],linewidth=5, color='m')
        
        loc13 = RK; loc14 = RA;
        k7, = ax.plot([loc13[0], loc14[0]], [loc13[1], loc14[1]], [loc13[2], loc14[2]],linewidth=5, color='m')
        
        pt = b
        k8, = ax.plot([pt[0]], [pt[1]], [pt[2]], 'ko', markersize=5, markerfacecolor='k')
        
        pt = rt
        k9, = ax.plot([pt[0]], [pt[1]], [pt[2]], 'ko', markersize=5, markerfacecolor='k')
        
        pt = rc
        k10, = ax.plot([pt[0]], [pt[1]], [pt[2]], 'ko', markersize=5, markerfacecolor='k')
        
        pt = lt
        k11, = ax.plot([pt[0]], [pt[1]], [pt[2]], 'ko', markersize=5, markerfacecolor='k')
        
        pt = lc
        k12, = ax.plot([pt[0]], [pt[1]], [pt[2]], 'ko', markersize=5, markerfacecolor='k')
        
        ax.set_xlim([0.0, 4.0])
        ax.set_ylim([-2.0, 2.0])
        ax.set_zlim([0.0, 2.0])
        
        plt.pause(params.pause)
        
        # if (i < (mm-1)):
        k1.remove(); k2.remove();
        k3.remove(); k4.remove();
        k5.remove(); k6.remove()
        k7.remove(); k8.remove()
        k9.remove(); k10.remove()
        k11.remove(); k12.remove()

    plt.pause(10)

def plot(t, z, Torque, params):
    
    #  [x, xd, y, yd, z, zd, phi, phid, theta, thetad, psi, psid, ....
    #  phi_lh, phi_lhd, theta_lh, theta_lhd, psi_lh, psi_lhd, theta_lk, theta_lkd, ...
    #  phi_rh, phi_rhd, theta_rh, theta_rhd, psi_rh, psi_rhd, theta_rk, theta_rkd]= getstate(Z);
    
    # plt.figure(2)
    # plt.subplot(2,1,1)
    # plt.plot(t, z[:,0], 'r', label=r'$x$')
    # plt.plot(t, z[:,2], 'b', label=r'$y$')
    # plt.plot(t, z[:,4], 'g', label=r'$z$')
    # plt.ylabel("body absolute positions")
    # plt.legend(loc=(1.0, 1.0), ncol=1, fontsize=7)
    
    # plt.subplot(2,1,2)
    # plt.plot(t,z[:,1],'r', label=r'$\dot{x}$')
    # plt.plot(t,z[:,3],'b', label=r'$\dot{y}$')
    # plt.plot(t,z[:,5],'g', label=r'$\dot{z}$')
    # plt.ylabel("body absolute rates")
    # plt.legend(loc=(1.0, 1.0), ncol=1, fontsize=7)
    # plt.xlabel('time')
    
    plt.figure(3)
    plt.subplot(2,1,1)
    # plt.plot(t,Torque[:,0],'r', label=r'$hip_{\phi}$')
    # plt.plot(t,Torque[:,1],'g', label=r'$hip_{\theta}$')
    # plt.plot(t,Torque[:,2],'b', label=r'$hip_{\psi}$')
    # plt.plot(t,Torque[:,3],'k', label=r'$knee_{\theta}$')
    plt.plot(Torque[:,0],'r', label=r'$hip_{\phi}$')
    plt.plot(Torque[:,1],'g', label=r'$hip_{\theta}$')
    plt.plot(Torque[:,2],'b', label=r'$hip_{\psi}$')
    plt.plot(Torque[:,3],'k', label=r'$knee_{\theta}$')
    plt.title('left leg')
    plt.ylabel("Torque")
    
    plt.subplot(2,1,2)
    # plt.plot(t,Torque[:,4],'r', label=r'$hip_{\phi}$')
    # plt.plot(t,Torque[:,5],'g', label=r'$hip_{\theta}$')
    # plt.plot(t,Torque[:,6],'b', label=r'$hip_{\psi}$')
    # plt.plot(t,Torque[:,7],'k', label=r'$knee_{\theta}$')
    plt.plot(Torque[:,4],'r', label=r'$hip_{\phi}$')
    plt.plot(Torque[:,5],'g', label=r'$hip_{\theta}$')
    plt.plot(Torque[:,6],'b', label=r'$hip_{\psi}$')
    plt.plot(Torque[:,7],'k', label=r'$knee_{\theta}$')
    plt.title('right leg')
    plt.ylabel("Torque")
    
    plt.show()
    
    
    # figure(7)
    # subplot(2,1,2);
    # plot(t,Torque(:,5),'r'); hold on;
    # plot(t,Torque(:,6),'g'); 
    # plot(t,Torque(:,7),'b'); 
    # plot(t,Torque(:,8),'k'); 
    # title('right leg')
    # ylabel('Torque');
    # legend('hip-phi','hip-theta','hip-psi','knee-theta');
    # xlabel('time');
    
    # % figure(2)
    # % subplot(2,1,1)
    # % plot(t,x,'r'); hold on;
    # % plot(t,y,'b');
    # % plot(t,z,'g');
    # % title('body absolute positions');
    # % subplot(2,1,2)
    # % plot(t,xd,'r'); hold on;
    # % plot(t,yd,'b');
    # % plot(t,zd,'g');
    # % title('body absolute rates');
    # % xlabel('time');
    # % legend('x','y','z');
    # % 
    # % figure(3)
    # % subplot(2,1,1)
    # % plot(t,phi,'r'); hold on;
    # % plot(t,theta,'b');
    # % plot(t,psi,'g');
    # % title('body absolute angles');
    # % subplot(2,1,2)
    # % plot(t,phid,'r'); hold on;
    # % plot(t,thetad,'b');
    # % plot(t,psid,'g');
    # % legend('phi','theta','psi');
    # % xlabel('time');
    # % title('body absolute rates');
    # % 
    # % figure(4)
    # % subplot(2,1,1)
    # % plot(t,phi_lh,'r'); hold on;
    # % plot(t,theta_lh,'b');
    # % plot(t,psi_lh,'g');
    # % plot(t,theta_lk,'k');
    # % title('left angles');
    # % subplot(2,1,2)
    # % plot(t,phi_lhd,'r'); hold on;
    # % plot(t,theta_lhd,'b');
    # % plot(t,psi_lhd,'g');
    # % plot(t,theta_lkd,'k');
    # % xlabel('time');
    # % legend('phi-hip','theta-hip','psi-hip','theta-knee');
    # % title('left rates');
    # % 
    # % figure(5)
    # % subplot(2,1,1)
    # % plot(t,phi_rh,'r'); hold on;
    # % plot(t,theta_rh,'b');
    # % plot(t,psi_rh,'g');
    # % plot(t,theta_rk,'k');
    # % title('right angles');
    # % subplot(2,1,2)
    # % plot(t,phi_rhd,'r'); hold on;
    # % plot(t,theta_rhd,'b');
    # % plot(t,psi_rhd,'g');
    # % plot(t,theta_rkd,'k');
    # % legend('phi-hip','theta-hip','psi-hip','theta-knee');
    # % title('right rates');
    # % xlabel('time');
    # % 
    # % figure(6)
    # % subplot(2,1,2)
    # % plot(t,P_RA_all(:,1),'r'); hold on
    # % plot(t,P_RA_all(:,2),'g'); 
    # % plot(t,P_RA_all(:,3),'b');
    # % title('right leg')
    # % ylabel('reaction force');
    # % subplot(2,1,1)
    # % plot(t,P_LA_all(:,1),'r'); hold on
    # % plot(t,P_LA_all(:,2),'g'); 
    # % plot(t,P_LA_all(:,3),'b'); 
    # % ylabel('reaction force');
    # % title('left leg')
    # % legend('x','y','z');
    # % xlabel('time');
    # % 


    
    return None