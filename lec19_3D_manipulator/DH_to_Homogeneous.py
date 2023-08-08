import sympy as sy

theta, d, a, alpha = sy.symbols('theta d a alpha', real=True)

H_z_theta = sy.Matrix(
    [
        [sy.cos(theta), -sy.sin(theta), 0, 0],
        [sy.sin(theta), sy.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

H_z_d = sy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])

H_x_a = sy.Matrix([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

H_x_alpha = sy.Matrix(
    [
        [1, 0, 0, 0],
        [0, sy.cos(alpha), -sy.sin(alpha), 0],
        [0, sy.sin(alpha), sy.cos(alpha), 0],
        [0, 0, 0, 1],
    ]
)

H = H_z_theta * H_z_d * H_x_a * H_x_alpha

m, n = H.shape

for i in range(m):
    for j in range(n):
        print(f'H[i,j] = {H[i,j]}')
