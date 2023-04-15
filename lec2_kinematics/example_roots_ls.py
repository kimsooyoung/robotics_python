from scipy.optimize import least_squares

# symbolic answer => -1, +2 
def func(x):
    return x**2 - x - 2

# state가 2개 이상이 되면 하나의 tuple로 묶어서 전달해야 함
def multi_var_func(var):
    x, y = var
    return x**2 - x - 2 + y**2 - y - 2

# parameter는 튜플로 묶이지 않고 각각 전달됨
def multi_var_func_w_params(var, radius, nothing):
    x, y = var
    return (x - 2)**2 + (y - 2)**2 - radius**2

res1 = least_squares(func, 0, bounds = ((-1), (0)))
print(res1.x)

res2 = least_squares(multi_var_func, (0, 0), bounds = ((-1), (0)))
print(res2.x)

radius = 3
nothing = 12.34
res3 = least_squares(
    multi_var_func_w_params, (0, 0), 
    bounds = ((-1, -1), (2, 2)),
    args=(radius, nothing)
)
print(res3.x)