import sympy as sy


def homogeneous(theta, L):
    
    H = sy.Matrix([
        [sy.cos(theta), -sy.sin(theta), L],
        [sy.sin(theta),  sy.cos(theta), 0],
        [0, 0, 1]
    ])

    return H
    
def revolute(theta, r, u):
    
    ux, uy, uz = u
    
    cth = sy.cos(theta)
    sth = sy.sin(theta)
    vth = 1 - cth
    
    R11 = ux**2 * vth + cth;  R12 = ux*uy*vth - uz*sth; R13 = ux*uz*vth + uy*sth
    R21 = ux*uy*vth + uz*sth; R22 = uy**2*vth + cth   ; R23 = uy*uz*vth - ux*sth
    R31 = ux*uz*vth - uy*sth; R32 = uy*uz*vth + ux*sth; R33 = uz**2*vth + cth
    
    R = sy.Matrix([
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33]
    ])

    I = sy.eye(3)
    
    T12 = (I-R)*r
    
    T = sy.Matrix([
        [R11, R12, R13, T12[0]],
        [R21, R22, R23, T12[1]],
        [R31, R32, R33, T12[2]],
        [0,   0,   0,   1]
    ])
    
    return T
    
def derive_nlink(dof, method):

    # dof = 3
    # dof = 2
    # method = "zeroref"
    # method = "homogenous"

    q = sy.symarray('q', dof)
    u = sy.symarray('u', dof)
    a = sy.symarray('a', dof)

    g = sy.symbols('g', real=True)

    m = sy.symarray('m', dof)
    c = sy.symarray('c', dof)
    l = sy.symarray('l', dof)
    I = sy.symarray('I', dof)

    if method == 'zeroref':
        r = sy.Matrix([0, 0, 0])
        pin = sy.Matrix([0, 0, 1])
        T01 = revolute(q[0], r, pin)

        T0 = [T01]
        
        dist_joint = 0
        for i in range(1, dof):
            dist_joint -= l[i-1]
            r = sy.Matrix([0, dist_joint, 0])  # joint i is at sum l(i) along -ive y-axis
            pin = sy.Matrix([0, 0, 1])  # axis along z
            T_temp = revolute(q[i], r, pin)
            T0.append(T0[i-1] * T_temp)  # e.g., H02 = H01 * H12


        dist_joint = 0
        r_G = []
        r_P = []
        for i in range(dof):
            r_G_i = sy.simplify(T0[i] * sy.Matrix([0, dist_joint - c[i], 0, 1]), seconds=10)
            dist_joint -= l[i]
            r_P_i = sy.simplify(T0[i] * sy.Matrix( [0, dist_joint, 0, 1] ), seconds=10)
            r_G.append(r_G_i)
            r_P.append(r_P_i)
            
            print(f'simplified position vector no {i+1}')

    elif method == "homogenous":
        # position vectors using homogenous transformations
        H01 = homogeneous(3 * sy.pi / 2 + q[0], 0)
        H0 = [H01]
        for i in range(1, dof):
            H_temp = homogeneous(q[i], l[i-1])
            H0.append(H0[i-1] * H_temp)  # e.g., H02 = H01 * H12

        r_G = []
        r_P = []
        for i in range(dof):
            r_G_i = sy.simplify(H0[i] * sy.Matrix([c[i], 0, 1]), seconds=10)
            r_P_i = sy.simplify(H0[i] * sy.Matrix([l[i], 0, 1]), seconds=10)
            r_G.append(r_G_i)
            r_P.append(r_P_i)
            
            print(f'simplified position vector no {i+1}')
    else:
        raise ValueError('method for derivation should be zeroref or homogenous')

    ### velocity vectors ###
    q_mat = sy.Matrix(q)
    u_mat = sy.Matrix(u)

    v_G = []
    for i in range(dof):
        r_G_i = sy.Matrix([r_G[i][0], r_G[i][1], r_G[i][2]])
        v_G_i = r_G_i.jacobian(q_mat) * sy.Matrix(u_mat)
        v_G.append(v_G_i)

    om = [u[0]]
    for i in range(1, dof):
        om_i = om[i-1] + u[i]
        om.append(om_i)

    ### Lagrangian ###
    T = 0
    V = 0
    for i in range(dof):
        v_G_i = sy.Matrix(v_G[i])
        T += 0.5 * m[i] * (v_G_i.dot(v_G_i)) + 0.5 * I[i] * om[i]**2
        V += m[i] * g * r_G[i][1]

    L = T - V

    ### Equations of Motion (EOM) ###
    dLdqdot = []
    ddt_dLdqdot = []
    dLdq = []
    EOM = []

    for ii in range(dof):
        dLdqdot_ii = sy.diff(L, u[ii])
        sum_val = 0

        for j in range(dof):
            sum_val += sy.diff(dLdqdot_ii, q[j]) * u[j] + \
                    sy.diff(dLdqdot_ii, u[j]) * a[j]

        dLdqdot.append(dLdqdot_ii)
        
        ddt_dLdqdot_ii = sum_val
        dLdq_ii = sy.diff(L, q[ii])

        ddt_dLdqdot.append(sum_val)
        dLdq.append(sy.diff(L, q[ii]))

        EOM.append(ddt_dLdqdot_ii - dLdq_ii)

        print(f'EOM (i = {ii+1}) done')

    EOM = sy.Matrix(EOM)
    a_mat = sy.Matrix(a)

    # test = list(zip(a_mat, [0]*dof))
    # print(test)

    M = EOM.jacobian(a_mat)
    b = EOM.subs([
        *list(zip(a_mat, [0]*dof)),
    ])
    G = b.subs([
        *list(zip(u_mat, [0]*dof)),
    ])
    C = b - G

    print('N, G, C done')


    with open("nlink_rhs.py", "w") as f:
        
        f.write("import numpy as np \n\n")
        f.write("def cos(angle): \n")
        f.write("    return np.cos(angle) \n\n")
        f.write("def sin(angle): \n")
        f.write("    return np.sin(angle) \n\n")

        f.write("def nlink_rhs(z, t, params): \n\n")
        
        for i in range(dof):
            f.write(f"    m_{i} = params.m{i+1}; I_{i} = params.I{i+1}\n    c_{i} = params.c{i+1}; l_{i} = params.l{i+1};\n")
        f.write("    g = params.g\n\n")
        
        for i in range(dof):
            f.write(f"    q_{i}, u_{i} = z[{2*i}], z[{2*i+1}] \n")
        f.write("\n")
        
        for i in range(dof):
            for j in range(dof):
                elem = sy.simplify(M[i,j])
                f.write(f"    M{i+1}{j+1} = {elem} \n\n")
        f.write("\n")
        
        for i in range(dof):
            elem = sy.simplify( C[i] )
            f.write(f"    C{i+1} = {elem} \n\n")
        f.write("\n")
        
        for i in range(dof):
            elem = sy.simplify( G[i] )
            f.write(f"    G{i+1} = {elem} \n\n")
        f.write("\n")
        
        f.write("    A = np.array([ \n")
        for i in range(dof):
            f.write(f"        [")
            for j in range(dof):
                f.write(f"M{i+1}{j+1}")
                if j != dof-1:
                    f.write(f", ")
            f.write(f"]")
            if i != dof-1:
                f.write(f",\n")
            else:
                f.write(f"\n")
        f.write("    ]) \n\n")
        
        f.write("    b = -np.array([ \n")
        for i in range(dof):
            f.write(f"        [C{i+1} + G{i+1}]")
            if i != dof-1:
                f.write(f",")
            f.write(f"\n")
        f.write("    ]) \n\n")
        
        f.write(f"    x = np.linalg.solve(A, b)\n\n")
        
        f.write("    output = np.array([\n")
        for i in range(dof):
            f.write(f"        u_{i}, x[{i},0]")
            if i != dof-1:
                f.write(f",")
            f.write(f"\n")
        f.write("    ])\n\n")
        
        f.write("    return output \n\n")
        print('nlink_rhs.py done')
        
    with open("nlink_animate.py", "w") as f:
        
        f.write("import numpy as np \n")
        f.write("import matplotlib.pyplot as plt\n\n")
        f.write("def cos(angle): \n")
        f.write("    return np.cos(angle) \n\n")
        f.write("def sin(angle): \n")
        f.write("    return np.sin(angle) \n\n")
        
        f.write("def nlink_animate(t_interp, z_interp, params): \n\n")
        
        for i in range(dof):
            f.write(f"    l_{i} = params.l{i+1} \n")
        f.write("\n")
        
        f.write("    ll = params.l1*4 + 0.2\n")
        
        f.write("    for i in range(0,len(t_interp)):\n")
        for i in range(dof):
            f.write(f"        q_{i} = z_interp[i,{2*i}]\n")
        f.write("\n")
        
        f.write("        P0 = np.array([0, 0])\n")
        for i in range(dof):
            f.write(f"        P{i+1} = np.array([{r_P[i][0]},{r_P[i][1]}])\n")
        f.write("\n")
        
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'black', 'yellow']
        for i in range(dof):
            f.write(f"        h{i+1}, = plt.plot([P{i}[0], P{i+1}[0]],[P{i}[1], P{i+1}[1]],linewidth=5, color='{colors[i-1]}')\n")
        f.write("\n")
        
        f.write("        plt.xlim([-ll, ll])\n")
        f.write("        plt.ylim([-ll, ll])\n")
        f.write("        plt.gca().set_aspect('equal')\n")
        f.write("\n")
        
        f.write("        plt.pause(params.pause)\n")
        f.write("\n")
        
        f.write("        if (i < len(t_interp)-1):\n")
        for i in range(dof):
            f.write(f"            h{i+1}.remove()\n")
        f.write("\n")
        
        f.write("    plt.show()\n\n")
        print('nlink_animate.py done')