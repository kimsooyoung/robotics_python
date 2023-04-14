import numpy as np
import matplotlib.pyplot as plt

from BezierCurve import BezierCurveMaker
from StanceCurve import StanceCurveMaker

def fullTraj(points, delta, num_sample=500):
    
    # Bezier Curve Maker
    bc_maker = BezierCurveMaker(len(points) - 1, points)
    # Stance Curve Maker    
    L = abs(points[0][0] - points[-1][0])
    y_offset = points[0][1]
    sc_maker = StanceCurveMaker(delta=delta, L=L, y_offset=y_offset)

    num_sample = num_sample
    sample_points = np.linspace(0, 1, num_sample)
    
    bc_result = np.zeros((num_sample, 2))
    for i in range(len(sample_points)):
        bc_result[i] = bc_maker.bezierPoint(sample_points[i])
        
    sc_result = np.zeros((num_sample, 2))
    for i in range(len(sample_points)):
        sc_result[i] = sc_maker.stancePoint(sample_points[i])
        
    full_traj = np.concatenate((bc_result, sc_result), axis=0)

    return full_traj

if __name__=="__main__":
    points = [
        [-170,  -470],
        [-242, -470],
        [-300, -360],
        [-300, -360],
        [-300, -360],
        [0, -360],
        [0, -360],
        [0, -320],
        [300, -320],
        [300, -320],
        [242, -470],
        [170, -470],
    ]
    
    traj = fullTraj(points, delta=50, num_sample=500)
    
    fig = plt.figure()

    for point in traj:
        plt.plot(point[0], point[1], color = 'black', marker = 'o', markersize=1)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-300,300)
    plt.ylim(-600,100)
    plt.grid()

    plt.show()