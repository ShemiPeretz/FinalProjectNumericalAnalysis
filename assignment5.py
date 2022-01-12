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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy



class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, hull, shape_area: np.float32):
        self.hull = hull
        self.shape_area = shape_area

    def area(self):
        print(f"Hull Average Area:{self.shape_area}")
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

        # replace these lines with your solution]
        start_time = time.time()
        points = []
        n = 0
        while (time.time() - start_time) < maxtime*0.3:
            points.append(sample())
            if n >= 4999:
                n += 1
                break
            else:
                n += 1

        clusters_start = 20
        clusters = clusters_start
        hulls = []
        volumes = []
        areas = []


        # while time.time() - start_time < maxtime*0.5:
        #     if c > 30:
        #         break
        #     else:
        #         kmeans.append(KMeans(n_clusters=c, init='k-means++', max_iter=300, n_init=10, random_state=0))
        #         kmeans[c-3].fit(points)
        #         distortions.append(kmeans[c-3].inertia_)
        #         c += 1
        #
        # # plot the cost against K values
        # plt.plot(range(len(distortions)), distortions, color='g', linewidth='3')
        # plt.xlabel("Value of K")
        # plt.ylabel("Squared Error (Cost)")
        # plt.show()  # clear the plot
        #
        # for i in range(len(distortions)-2):
        #     pass


        while (time.time() - start_time) < maxtime*0.8:
            if (clusters - clusters_start) > 10:
                break
            else:
                kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
                kmeans.fit(points)
                points_to_fit = kmeans.cluster_centers_
                hull = scipy.spatial.ConvexHull(points_to_fit)
                hulls.append(hull)
                areas.append(hull.area / 2)
                # areas.append(hull.area)
                volumes.append(hull.volume)
                clusters += 1

        avg_area = 0
        no_areas = 0
        for i in range(len(areas)-2):
            if abs(areas[i] - areas[i+1]) <= 0.1:
                avg_area += areas[i]
                no_areas += 1
        avg_area = avg_area/no_areas

        avg_volume = 0
        no_volumes = 0
        for i in range(len(volumes)-2):
            if abs(volumes[i] - volumes[i+1]) <= 0.1:
                avg_volume += volumes[i]
                no_volumes += 1
        avg_volume = avg_volume/no_volumes

        # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', zorder=5)
        # plt.show()
        #
        # sorted_points_to_fit = sorted(points_to_fit, key=lambda k: [k[1], k[0]])
        # radios = (sorted_points_to_fit[clusters-1][1] - sorted_points_to_fit[0][1])/2
        # plt.plot(points_to_fit[hull.vertices, 0], points_to_fit[hull.vertices, 1], 'r', lw=2)
        # plt.plot(points_to_fit[hull.vertices[0], 0], points_to_fit[hull.vertices[0], 1], 'ro')
        # plt.show()

        result = MyShape(hulls[4], np.float32(avg_area))
        # print(f"time for clustering and hull: {time.time() - T}")
        # print(f"Total time: {time.time() - start_time}")
        print(f"pi ={np.pi}")
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

    def shoelace_area(self, points, n):
        x_list = [points[i][0] for i in range(n)]
        y_list = [points[i][1] for i in range(n)]
        a1, a2 = 0, 0
        x_list.append(x_list[0])
        y_list.append(y_list[0])
        for j in range(len(x_list) - 1):
            a1 += x_list[j] * y_list[j + 1]
            a2 += y_list[j] * x_list[j + 1]
        l = abs(a1 - a2) / 2
        return l

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
        print(abs(a - np.pi))
        self.assertLess(abs(a - np.pi), 0.01)
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
