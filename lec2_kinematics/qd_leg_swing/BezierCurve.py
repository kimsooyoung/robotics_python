import numpy as np
import matplotlib.pyplot as plt

def factorial(num):
    output = 1

    if num == 0:
        return output
    elif num == 1:
        return output
    else: 
        output = output * num * factorial(num - 1)

    return output

def binomialFactor(n, i):
    return factorial(n) / (factorial(i) * factorial(n - i))


class BezierCurveMaker(object):
        
    """
    Assume that points will have "numpy.ndarray" type
    """
    def __init__(self, order, points):
        super().__init__()

        self.order = order

        if type(points) == np.ndarray:
            self.points = points
        elif type(points) == list: 
            self.points = np.array(points)

    def bezierPoint(self, t):
        # Caution!! array elements must be float
        m, n = self.points.shape
        output = np.zeros(n)

        for i in range(len(self.points)):
            for j in range(len(self.points[i])):
                output[j] += binomialFactor(self.order, i) * pow(t, i) * pow(1-t, self.order-i) * self.points[i][j]

        return output
    
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
    
    num_sample = 500
    sample_points = np.linspace(0, 1, num_sample)
    
    bc_maker = BezierCurveMaker(11, points)

    bc_result = np.zeros((num_sample, 2))
    for i in range(len(sample_points)):
        bc_result[i] = bc_maker.bezierPoint(sample_points[i])

    fig = plt.figure()

    for point in bc_result:
        plt.plot(point[0], point[1], color = 'black', marker = 'o', markersize=1)
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-300,300)
    plt.ylim(-600,100)
    plt.grid()

    plt.show()