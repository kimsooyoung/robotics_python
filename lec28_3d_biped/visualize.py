import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt
from scipy import interpolate

from joint_locations import joint_locations

def animate(t_all, z_all, params, fps, view):
    
    l0, l1, l2 = params.l0, params.l1, params.l2
    w = params.w
    
    m, n = z_all.shape
    
    z_all_plot = np.zeros((m, n/2))
    for i in range(0, n, 2):
        z_all_plot[:, i/2] = z_all[:, i]
    
    total_frames = round(t_all[-1] * fps)
    zz = np.zeros((total_frames, n/2))
    t = np.arange(0, t_all[-1], total_frames)
    
    for i in range(n/2):
        f = interpolate.interp1d(t_all, z_all_plot[:, i])
        zz[:, i] = f(t)
    
    mm, nn = zz.shape
    
    fig = plt.figure(1)
    
    # For MacOS Users
    ax = p3.Axes3D(fig)

    # For Windows/Linux Users
    # ax = fig.add_subplot(111, projection='3d')
    
    for i in range(mm):
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
        
        loc1 = B; loc2 = LH;
        k2, = ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], [loc1[2], loc2[2]],linewidth=5, color='b')

        loc1 = LH; loc2 = RH;
        k3, = ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], [loc1[2], loc2[2]],linewidth=5, color='b')
    
        loc1 = LH; loc2 = LK;
        k4, = ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], [loc1[2], loc2[2]],linewidth=5, color='c')
        
        loc1 = RH; loc2 = RK;
        k5, = ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], [loc1[2], loc2[2]],linewidth=5, color='c')
        
        loc1 = LK; loc2 = LA;
        k6, = ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], [loc1[2], loc2[2]],linewidth=5, color='m')
        
        loc1 = RK; loc2 = RA;
        k7, = ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], [loc1[2], loc2[2]],linewidth=5, color='m')
        
        pt = b
        k8, = ax.plot([pt[0]], [pt[1]], [pt[2]], 'ko', markersize=20, markerfacecolor='k')
        
        pt = rt
        k9, = ax.plot([pt[0]], [pt[1]], [pt[2]], 'ko', markersize=10, markerfacecolor='k')
        
        pt = rc
        k10, = ax.plot([pt[0]], [pt[1]], [pt[2]], 'ko', markersize=10, markerfacecolor='k')
        
        pt = lt
        k11, = ax.plot([pt[0]], [pt[1]], [pt[2]], 'ko', markersize=10, markerfacecolor='k')
        
        pt = lc
        k12, = ax.plot([pt[0]], [pt[1]], [pt[2]], 'ko', markersize=10, markerfacecolor='k')
        
        ax.set_xlim([0.0, 4.0])
        ax.set_ylim([-2.0, 2.0])
        ax.set_zlim([0.0, 2.0])
        
        plt.pause(params.pause)
        
        if (i < (mm-1)):
            k1.remove(); k2.remove();
            k3.remove(); k4.remove();
            k5.remove(); k6.remove()
            k7.remove(); k8.remove()
            k9.remove(); k10.remove()
            k11.remove(); k12.remove()

    plt.show()

def plot():
    
    return None