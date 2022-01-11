"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from functionUtils import AbstractShape
import assignment4
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, hull, radius: np.float32, shape_area: np.float32):
        super(MyShape, self).__init__(hull, radius, shape_area)

    def area(self):
        return self.shape_area

class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        n = maxerr*1000000
        points = contour(n)
        area = self.shoelace_formula(points, n)

        return np.float32(area)
    
    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        # replace these lines with your solution
        # TODO: add time restriction
        n = 2000
        points = []

        for i in range(n-1):
            points.append(sample())
        x_values = [points[i][0] for i in range(0, n-1)]
        y_values = [points[i][1] for i in range(0, n-1)]

        clusters = 20
        kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=500, n_init=10, random_state=0)
        kmeans.fit(points)
        # # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', zorder=5)
        # # # plt.plot(x_values, y_values, 'bo', zorder=10)
        # plt.show()

        points_to_fit = kmeans.cluster_centers_
        sorted_points_to_fit = sorted(points_to_fit, key=lambda k: [k[1], k[0]])
        hull = ConvexHull(points_to_fit)
        radios = (sorted_points_to_fit[clusters-1][1] - sorted_points_to_fit[0][1])/2
        area = hull.area
        # plt.plot(points_to_fit[hull.vertices, 0], points_to_fit[hull.vertices, 1], 'r--', lw=2)
        # plt.plot(points_to_fit[hull.vertices[0], 0], points_to_fit[hull.vertices[0], 1], 'ro')
        # plt.show()

        result = MyShape(hull, np.float32(radios), np.float32(area))

        return result

    def trapezoidal_area(self, x, y1, y2):
        return ((y1+y2)*x)/2

    def trapeziodal_rule(self, x_values, y_values, n):
        h = (x_values[n] - x_values[0]) / n

    def shoelace_formula(self, points, n):
        A = 0
        for i in range(n-1):
            A += points[i][0]*points[i+1][1]
        A += points[n][0]*points[1][1]
        for i in range(n-1):
            A -= points[i+1][0]*points[i][1]
        A -= points[1][0]*points[n][1]

        return abs(A)/2

    def best_k(self, x_values):
        wcss = []
        for i in range(1, 5):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(x_values)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()



##########################################################################


import unittest
from sampleFunctions import *
import functionUtils
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - 2*np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    # def test_5(self):
    #     ass5 = Assignment5()
    #     # ass5.area(contour)

if __name__ == "__main__":
    unittest.main()
