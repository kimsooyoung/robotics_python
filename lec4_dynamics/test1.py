from scipy.integrate import odeint
import numba
import timeit

def rober(u,t, k1, k2, k3):

    y1, y2, y3 = u

    dy1 = -k1*y1+k3*y2*y3
    dy2 =  k1*y1-k2*y2*y2-k3*y2*y3
    dy3 =  k2*y2*y2

    return [dy1,dy2,dy3]

u0  = [1.0,0.0,0.0]
t = [0.0, 1e5]
numba_f = numba.jit(rober,nopython=True)

def time_func():
    sol = odeint( numba_f, u0, t, args=(0.04, 3e7, 1e4), 
                 rtol=1e-3, atol=1e-6)

result = timeit.Timer(time_func).timeit(number=100)/100 # 0.0002520930000173394 seconds
print(result)

# 0.0006055544499999988