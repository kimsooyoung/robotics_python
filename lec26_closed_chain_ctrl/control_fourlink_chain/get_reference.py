import numpy as np
from scipy.optimize import fsolve

from visualize import animate
from quinticpolytraj import quinticpolytraj

def cos(angle):
    return np.cos(angle)

def sin(angle):
    return np.sin(angle);

def position_kinematics(z, params, l, alpha):
    
    l1, l2, l3, l4 = params.l1, params.l2, params.l3, params.l4
    lx, ly = params.lx, params.ly
    
    q1, q2, q3, q4 = z
    
    del_x = l1*sin(q1) + l2*sin(q1 + q2) - l3*sin(q3) - l4*sin(q3 + q4) - lx
    del_y = -l1*cos(q1) - l2*cos(q1 + q2) + l3*cos(q3) + l4*cos(q3 + q4) - ly
    
    leg_length = np.sqrt(l1**2 + 2*l1*l2*cos(q2) + l2**2)
    leg_angle = 0.5*q1 + 0.5*q3
    
    return del_x, del_y, leg_length - l, leg_angle - alpha

def get_reference(params):
    ### Solve q's such that end of final link is at lx,ly ###
    q1, q2, q3, q4 = -np.pi/3, np.pi/2, 0, 0
    
    ### kinematics initial condition ###
    q0 = [q1, q2, q3, q4]
    fsolve_params = (params, params.l_init, params.alpha_init)
    q_ini = fsolve(position_kinematics, q0, fsolve_params)
    q1, q2, q3, q4 = q_ini
    print(f"[initial condition] q1: {q1}, q2: {q2}, q3: {q3}, q4: {q4}")
    
    ### kinematics middle condition ###
    q0 = [q1, q2, q3, q4]
    fsolve_params = (params, params.l_mid, params.alpha_mid)
    q_middle = fsolve(position_kinematics, q0, fsolve_params)
    q1, q2, q3, q4 = q_middle
    print(f"[middle condition] q1: {q1}, q2: {q2}, q3: {q3}, q4: {q4}")
    
    ### kinematics final condition ###
    q0 = [q1, q2, q3, q4]
    fsolve_params = (params, params.l_final, params.alpha_final)
    q_final = fsolve(position_kinematics, q0, fsolve_params)
    q1, q2, q3, q4 = q_final
    print(f"[final condition] q1: {q1}, q2: {q2}, q3: {q3}, q4: {q4}")
    
    if params.show_phase:
        z_ini = [ q_ini[0], 0, q_ini[1], 0, q_ini[2], 0, q_ini[3], 0 ]
        animate([0], np.reshape(z_ini, (1,8)), params)
    
        z_middle = [ q_middle[0], 0, q_middle[1], 0, q_middle[2], 0, q_middle[3], 0 ]
        animate([0], np.reshape(z_middle, (1,8)), params)
    
        z_final = [ q_final[0], 0, q_final[1], 0, q_final[2], 0, q_final[3], 0 ]
        animate([0], np.reshape(z_final, (1,8)), params)
    
    ### Traj generation ###
    t_init, t_mid, t_end = 0, params.t_end/2, params.t_end
    
    q1_ref, q1d_ref, q1dd_ref, t_ref = quinticpolytraj(q_ini[0], q_middle[0], q_final[0], t_init, t_mid, t_end)
    
    if params.leg == "minitaur" or params.leg == "atrias":
        q3_ref, q3d_ref, q3dd_ref, t_ref = quinticpolytraj(q_ini[2], q_middle[2], q_final[2], t_init, t_mid, t_end)
    elif params.leg == "digit":
        q3_ref, q3d_ref, q3dd_ref, t_ref = quinticpolytraj(q_ini[3], q_middle[3], q_final[3], t_init, t_mid, t_end)
    
    q1_refs = np.column_stack([q1_ref, q1d_ref, q1dd_ref])
    q3_refs = np.column_stack([q3_ref, q3d_ref, q3dd_ref])
    
    return q_ini, t_ref, q1_refs, q3_refs