import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.optimize import fsolve

class parameters:
    def __init__(self):
        self.d1 = 1.3
        self.d2 = 1.4
        self.d4 = 0.9
        self.d5 = 0
        self.d6 = 0.4
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a5 = 0;
        self.a6 = 0
        self.alpha1 = -np.pi/2
        self.alpha2 = np.pi/2
        self.alpha3 = 0
        self.alpha4 = -np.pi/2
        self.alpha5 = np.pi/2
        self.alpha6 = 0
        self.theta3 = -np.pi/2
        self.pause = 0.05
        self.fps = 30

def cos(angle):
    return np.cos(angle)

def sin(angle):
    return np.sin(angle)

def DH(a,alpha,d,theta):

    cth = cos(theta);
    sth = sin(theta);
    cal = cos(alpha);
    sal = sin(alpha);

    H_z_theta = np.array([
                   [cth, -sth, 0, 0],
                   [sth,  cth, 0, 0],
                   [0 ,    0, 1, 0],
                   [0 ,    0, 0, 1]]);

    H_z_d = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, d],
                [0, 0, 0, 1]]);

    H_x_a = np.array([
                    [1, 0, 0, a],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]);

    H_x_alpha = np.array([
                  [1,   0,   0,   0],
                  [0,  cal, -sal, 0],
                  [0,  sal,  cal, 0],
                  [0,   0,    0,  1]]);

    H_z = np.matmul(H_z_theta,H_z_d);
    H_x = np.matmul(H_x_a,H_x_alpha);
    H = np.matmul(H_z,H_x);
    # H = H_z_theta*H_z_d*H_x_a*H_x_alpha;

    return H

def inverse_kinematics(X,Xdes):

    theta1 = X[0];
    theta2 = X[1]
    d3 = X[2];
    theta4 = X[3]
    theta5 = X[4]
    theta6 = X[5]

    parms = parameters();

    # Xdes = np.array([x_des,y_des,z_des,phi_des,theta_des,psi_des]);
    i = 0;
    x_des = Xdes[i]; i+=1;
    y_des = Xdes[i]; i+=1;
    z_des = Xdes[i]; i+=1;
    phi_des=Xdes[i]; i+=1;
    theta_des=Xdes[i]; i+=1;
    psi_des=Xdes[i]; i+=1;


    a1 = parms.a1; alpha1 = parms.alpha1; d1=parms.d1;
    H01 = DH(a1,alpha1,d1,theta1); #H^0_1

    a2 = parms.a2; alpha2 = parms.alpha2; d2=parms.d2;
    H12 = DH(a2,alpha2,d2,theta2); #H^1_2
    #
    a3 = parms.a3; alpha3 = parms.alpha3; theta3 = parms.theta3;
    H23 = DH(a3,alpha3,d3,theta3); #H^2_3

    a4 = parms.a4; alpha4 = parms.alpha4; d4=parms.d4;
    H34 = DH(a4,alpha4,d4,theta4); #H^1_2

    a5 = parms.a5; alpha5 = parms.alpha5; d5=parms.d5;
    H45 = DH(a5,alpha5,d5,theta5); #H^1_2

    a6 = parms.a6; alpha6 = parms.alpha6; d6=parms.d6;
    H56 = DH(a6,alpha6,d6,theta6); #H^1_2

    H02 = np.matmul(H01,H12);
    H03 = np.matmul(H02,H23);
    H04 = np.matmul(H03,H34);
    H05 = np.matmul(H04,H45);
    H06 = np.matmul(H05,H56);

    endOfLink6 = H06[0:3,3];
    R = H06[0:3,0:3];

    theta=np.arcsin(-R[2,0]);
    phi=np.arcsin(R[2,1]/cos(theta));
    psi=np.arcsin(R[1,0]/cos(theta));

    return endOfLink6[0]-x_des,endOfLink6[1]-y_des,endOfLink6[2]-z_des, \
           phi-phi_des,theta-theta_des,psi - psi_des


def animate(X,Xdes,order,title):
    theta1 = X[0];
    theta2 = X[1]
    d3 = X[2];
    theta4 = X[3]
    theta5 = X[4]
    theta6 = X[5]

    parms = parameters()

    a1 = parms.a1; alpha1 = parms.alpha1; d1=parms.d1;
    H01 = DH(a1,alpha1,d1,theta1); #H^0_1

    a2 = parms.a2; alpha2 = parms.alpha2; d2=parms.d2;
    H12 = DH(a2,alpha2,d2,theta2); #H^1_2
    #
    a3 = parms.a3; alpha3 = parms.alpha3; theta3 = parms.theta3;
    H23 = DH(a3,alpha3,d3,theta3); #H^2_3

    a4 = parms.a4; alpha4 = parms.alpha4; d4=parms.d4;
    H34 = DH(a4,alpha4,d4,theta4); #H^1_2

    a5 = parms.a5; alpha5 = parms.alpha5; d5=parms.d5;
    H45 = DH(a5,alpha5,d5,theta5); #H^1_2

    a6 = parms.a6; alpha6 = parms.alpha6; d6=parms.d6;
    H56 = DH(a6,alpha6,d6,theta6); #H^1_2
    #
    #%Location of joint 1
    endOfLink1 = H01[0:3,3];
    # print(endOfLink1)
    #
    # #Location of joint 2
    H02 = np.matmul(H01,H12);
    endOfLink2 = H02[0:3,3];
    # print(endOfLink2)
    #
    #Location of joint 3
    H03 = np.matmul(H02,H23); #H01*H12*H23;
    endOfLink3 = H03[0:3,3];
    # # print(endOfLink3)

    #Location of joint 4
    H04 = np.matmul(H03,H34); #H01*H12*H23;
    endOfLink4 = H04[0:3,3];
    # # print(endOfLink3)

    #Location of joint 5
    H05 = np.matmul(H04,H45); #H01*H12*H23;
    endOfLink5 = H05[0:3,3];
    # # print(endOfLink3)

    #Location of joint 5
    H06 = np.matmul(H05,H56); #H01*H12*H23;
    endOfLink6 = H06[0:3,3];
    # # print(endOfLink3)

    #
    # #end-effector position and orientation.
    position_of_end_effector = H06[0:3,3];
    orientation_of_end_effector = H06[0:3,0:3];
    # print(position_of_end_effector)
    # print(orientation_of_end_effector)
    
    ax = fig.add_subplot(order, projection='3d')
    # ax = p3.Axes3D(fig)
    
    ax.set_title(title)

    line1, = ax.plot([0, endOfLink1[0]],[0, endOfLink1[1]],[0, endOfLink1[2]], color='red', linewidth=2)
    line2, = ax.plot([endOfLink1[0], endOfLink2[0]],[endOfLink1[1], endOfLink2[1]],[endOfLink1[2], endOfLink2[2]],
                      color='blue', linewidth=2)
    line3, = ax.plot([endOfLink2[0], endOfLink3[0]],[endOfLink2[1], endOfLink3[1]],[endOfLink2[2], endOfLink3[2]],
                      color='green', linewidth=2)
    line4, = ax.plot([endOfLink3[0], endOfLink4[0]],[endOfLink3[1], endOfLink4[1]],[endOfLink3[2], endOfLink4[2]],
                      color='yellow', linewidth=2)
    line5, = ax.plot([endOfLink4[0], endOfLink5[0]],[endOfLink4[1], endOfLink5[1]],[endOfLink4[2], endOfLink5[2]],
                      color=np.array([255/255, 165/255,0]), linewidth=2)
    line6, = ax.plot([endOfLink5[0], endOfLink6[0]],[endOfLink5[1], endOfLink6[1]],[endOfLink5[2], endOfLink6[2]],
                          color='red', linewidth=2)
    point = ax.plot(Xdes[0],Xdes[1],Xdes[2],'ko')
    
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    ax.view_init(azim=63,elev=14)

if __name__=="__main__":
    parms = parameters();

    #initial guess
    theta1, theta2, d3 = 0, 0, 0
    theta4, theta5, theta6 = 1, -1, 1
    
    x_des, y_des, z_des = 1.5, 1.5, 2
    phi_des, theta_des, psi_des = 0, 0, 0
    
    Xdes = np.array([x_des,y_des,z_des,phi_des,theta_des,psi_des])
    X0 = np.array([theta1, theta2, d3, theta4, theta5, theta6])

    fig = plt.figure(figsize = (16, 8))
    animate(X0, Xdes, 121, "Initial Guess")

    #X0 = np.array([0.064624220597635 ,  1.695634896738484 ,  0.706237840485306,   1.570796326794651,  -1.695634896731934,  -0.064624220594252]);
    X = fsolve(inverse_kinematics, X0,Xdes,maxfev=500,xtol=1e-6)
    FVAL = inverse_kinematics(X,Xdes)
    print(f"IK result : {X}")
    print(f"FVAL : {FVAL}")
    
    animate(X, Xdes, 122, "Final Solution")
    plt.show()
    