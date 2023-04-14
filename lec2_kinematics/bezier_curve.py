import numpy as np

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
    def __init__(self, n, points):
        super().__init__()

        self.n = n

        if type(points) == np.ndarray:
            self.points = points
        elif type(points) == list: 
            self.points = np.array(points)

    def getBezierPoint(self, t):
        # Caution!! array elements must be float
        output = np.array([0.0, 0.0])

        for i in range(len(self.points)):
            output[0] += binomialFactor(self.n, i) * pow(t, i) * pow(1-t, self.n-i) * self.points[i][0]
            output[1] += binomialFactor(self.n, i) * pow(t, i) * pow(1-t, self.n-i) * self.points[i][1]

        return output