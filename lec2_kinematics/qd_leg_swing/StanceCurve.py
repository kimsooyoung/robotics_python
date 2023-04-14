import numpy as np
import matplotlib.pyplot as plt


class StanceCurveMaker(object):
    def __init__(self, delta = 0.05, L=5, y_offset=0, dimen=2):
        super().__init__()

        self.dimen = dimen
        self.delta = delta
        self.L = L
        self.y_offset = y_offset

    """
    t = 0 will be (L/2, 0), and
    t = 1 will be (-L/2, 0)

    And middle point's value will be (0, -delta)
    """
        
    def stancePoint(self, t): 
        output = np.zeros(self.dimen)
        
        # TODO: 3D stance curve
        output[0] = self.L * (0.5 - t)
        output[1] = self.y_offset - self.delta * np.cos(np.pi * (0.5 - t))

        return output

if __name__=="__main__":
    
    num_sample = 500
    sample_points = np.linspace(0, 1, num_sample)
    
    sc_maker = StanceCurveMaker(delta=50, L=340, y_offset=-470)
    
    sc_result = np.zeros((num_sample, 2))
    for i in range(len(sample_points)):
        sc_result[i] = sc_maker.stancePoint(sample_points[i])
        
    fig = plt.figure()

    for point in sc_result:
        plt.plot(point[0], point[1], color = 'black', marker = 'o', markersize=1)
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-300,300)
    plt.ylim(-600,100)
    plt.grid()

    plt.show()