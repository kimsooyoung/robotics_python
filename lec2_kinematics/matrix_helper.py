import numpy as np


def calc_rot_2d(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])


def calc_homogeneous_2d(theta, trans):
    output = np.identity(3)
    output[:2, :2] = calc_rot_2d(theta)
    output[:2, 2] = np.transpose(np.array(trans))

    return output


if __name__ == '__main__':
    trans = [30.0, 1.0]
    print(calc_homogeneous_2d(0.0, trans))
