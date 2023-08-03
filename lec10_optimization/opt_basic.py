import scipy.optimize as opt


def cost(param):
    x1, x2 = param

    return 100 * (x2 - x1)**2 + (1 - x1)**2


initial_val = [0, 0]

# result = opt.minimize(cost, initial_val, method='BFGS')
result = opt.minimize(cost, initial_val, method='CG')
print(result)
print(result.x)
