# ref from : https://www.youtube.com/watch?v=JGk1jsAomDk

# Linear Regression
import numpy as np
import casadi as ca
import casadi.tools as ca_tools
import matplotlib.pyplot as plt

x = [0,45,90,135,180]
y = [667,661,757,871,1210]

m = ca.SX.sym('m')
c = ca.SX.sym('c')

obj = 0

for i in range(len(x)):
    obj += (y[i] - m*x[i] - c)**2

lbx = ca.DM([-np.inf, -np.inf])
ubx = ca.DM([np.inf, np.inf])
lbg = ca.DM([-np.inf])
ubg = ca.DM([np.inf])

prob_struct = { 'f': obj, 'x': ca.vertcat(m, c) }
opts_setting = {
    'ipopt.max_iter':100, 
    'ipopt.print_level':0, 
    'print_time':0, 
    'ipopt.acceptable_tol':1e-8, 
    'ipopt.acceptable_obj_change_tol':1e-6
}

solver = ca.nlpsol('solver', 'ipopt', prob_struct, opts_setting)

sol = solver(x0=ca.DM([0.5, 1]), lbx=ca.DM([-np.inf, -np.inf]), ubx=ca.DM([np.inf, np.inf]))
m_opt = sol['x'][0]
c_opt = sol['x'][1]
print(f"Optimal solution: {sol['x']}")
print(f"Optimal cost: {sol['f']}")

# 2D Visualization
def viz_plot(m, c):
    
    x = np.linspace(0, 180, 100)
    y = m*x + c

    plt.figure(1)
    plt.plot(x, y, linewidth=5, markersize=5)
    plt.plot([0,45,90,135,180], [667,661,757,871,1210], 'bo', markersize=10)
    plt.show()

# 3D Visualization with ca.Function()
def viz_mesh(m, c, obj):

    obj_fun = ca.Function('obj_fun', [m, c], [obj])

    m_range = np.linspace(-1, 6, 100)
    c_range = np.linspace(400, 800, 100)

    m_mesh, c_mesh = np.meshgrid(m_range, c_range)
    f_mesh = np.zeros((100, 100))

    for i in range(100):
        for j in range(100):
            f_mesh[i, j] = obj_fun(m_mesh[i, j], c_mesh[i, j])

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(m_mesh, c_mesh, f_mesh, cmap='coolwarm')
    ax.set_xlabel('m')
    ax.set_ylabel('c')
    ax.set_zlabel('f(m, c)')
    ax.set_title('Objective function')
    plt.show()

viz_plot(m_opt, c_opt)
viz_mesh(m, c, obj)