import sympy as sy

def calc_rot_2d(theta):
    return sy.Matrix([
        [sy.cos(theta), -sy.sin(theta)],
        [sy.sin(theta),  sy.cos(theta)],
    ])

def calc_homogeneous_2d(trans_x, trans_y, theta):
    output = sy.eye(3)
    output[:2, :2] = calc_rot_2d(theta)
    output[0, 2] = trans_x
    output[1, 2] = trans_y
    
    return output
    
if __name__=="__main__":
    # vector mul example
    h_p = sy.Matrix([2, 0, 1])
    H_test = calc_homogeneous_2d(0, 0, sy.pi / 2)
    print(H_test * h_p)


    cos1 = sy.cos(3*sy.pi/2 + 12)
    print(cos1)


    print(calc_homogeneous_2d(30, 1, sy.pi / 4))