import sympy as sy
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

x, y, z = sy.symbols('x y z')

a = sy.Matrix([
    [x],
    [y],
    [z],
    [0]
])

print(a.dot(a))
print(a.shape)

ll = [1,2,3,4]

lm = sy.Matrix(ll)

print(lm)

X_des = np.zeros((8,1))

for i in range(8):
    X_des[i] = i
    
print(X_des)

k = np.array([1,2,3])
k = np.hstack(( np.zeros(6) , k))
print(k)

t = np.zeros( (1,3) )
t[0] = [1,2,3] 
print(t)

t = np.vstack( (t, np.array([ 0, 0, 0 ]) ) )
t = np.vstack( (t, np.array([ 0.1, 0.2, 0.3 ]) ) )
print(t)
print(t[:,0])

t_temp = np.zeros( (1,1) )
t_temp[0] = 0.0
test_arr = np.zeros

M = np.array([
    [1,2,3],
    [2,3,4]
])

print(M.shape)


fig = plt.figure(1)

# # For MacOS Users
# ax = p3.Axes3D(fig)

# For Windows/Linux Users

for i in range(11):
    ax = fig.add_subplot(111, projection='3d')

    loc1 = np.array([0,0,0])
    loc2 = np.array([1,1,1])

    k1, = ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], [loc1[2], loc2[2]],linewidth=5, color='r')

    pt = np.array([i*0.1,i*0.1,i*0.1])
    k8, = ax.plot([pt[0]], [pt[1]], [pt[2]], 'ko', markersize=20, markerfacecolor='k')

    # Ubuntu and Windows
    # plt.pause(0.01)
    
    # MacOS
    plt.pause(0.01)
    
    k1.remove()
    k8.remove()

plt.pause(10)

# plt.close()
