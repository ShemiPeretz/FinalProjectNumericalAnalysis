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
from sklearn.cluster import MiniBatchKMeans,DBSCAN
from collections import Counter
import scipy



class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, shape_area):
        self.shape_area = shape_area

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

        # replace these lines with your solution]
        start_time = time.time()
        points = []
        n = 0
        while (time.time() - start_time) < maxtime*0.3:
            points.append(sample())
            if n >= 2500:
                n += 1
                break
            else:
                n += 1

        num_of_points = len(points)
        points_freq = DBSCAN(eps=0.1, min_samples=10).fit(points)
        labels = points_freq.labels_
        most_pop = self.popular_points(labels)
        new_points = [points[i] for i in range(num_of_points) if labels[i] in most_pop]

        clusters = 3
        areas = []
        k = []
        while (time.time() - start_time) < maxtime*0.8:
            if clusters > 36:
                break
            else:
                kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=0, batch_size=100)
                kmeans.fit(new_points)
                points_to_fit = kmeans.cluster_centers_
                hull = scipy.spatial.ConvexHull(points_to_fit)
                areas.append(hull.volume)
                k.append(clusters)
                clusters += 3

        if (time.time() - start_time) > 0.90 * maxtime:
            return MyShape(np.float32(np.average(areas)))

        ak_poly = np.polyfit(k, areas, deg=3)
        a_for_k = lambda x: ak_poly[0]*x**3 + ak_poly[1]*x**2 + ak_poly[2]*x + ak_poly[3]
        derivative = 0
        optimum_k = 0
        delta_der = 0

        # for i in range(len(k)):
        #     if i == 0:
        #         derivative = scipy.misc.derivative(a_for_k, k[i], dx=1e-6)
        #     else:
        #         new_derivative = scipy.misc.derivative(a_for_k, k[i], dx=1e-6)
        #         if i == 1:
        #             delta_der = abs(new_derivative - derivative)
        #             derivative = new_derivative
        #         elif abs(new_derivative - derivative) <= delta_der:
        #             optimum_k = i
        #             delta_der = abs(new_derivative - derivative)
        #             derivative = new_derivative
        #         else:
        #             derivative = new_derivative

        deriv_list = []
        for i in range(len(k)):
            deriv_list.append(scipy.misc.derivative(a_for_k, k[i], dx=1e-6))
        optimum_k = deriv_list.index(min(deriv_list))

        if (time.time() - start_time) > 0.95 * maxtime:
            return MyShape(areas[optimum_k])

        optimum_k -= 2
        best_k = []
        best_k_areas = []
        for i in range(3):
            if 0 <= (optimum_k + i) <= len(k):
                best_k.append(k[optimum_k + i])
            if 0 <= optimum_k + i <= len(k):
                best_k_areas.append(areas[optimum_k + i])

        best_poly = np.polyfit(best_k, best_k_areas, deg=3)
        best_poly_func = np.poly1d(best_poly)
        sum_area = 0

        points_for_sample = np.linspace(best_k[0], best_k[-1], 100)
        for point in points_for_sample:
            sum_area += best_poly_func(point)
        avg_area = sum_area/100


        result = MyShape(avg_area)

        return result

    def popular_points(self, val_lst):
        count = Counter(val_lst)
        times = []
        for item in count.items():
            no = item[1]
            if no/len(val_lst) >= 0.1:
                times.append(item[0])
        return times

    def clear_noise(self, points):
        num_of_points = len(points)
        points1 = DBSCAN(eps=0.1,min_samples=10).fit(points)
        labels = points1.labels_
        most_frequent_labels = self.popular_points(labels)
        points = [points[i] for i in range(num_of_points) if labels[i] in most_frequent_labels]
        return points

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
