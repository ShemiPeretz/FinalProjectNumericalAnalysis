"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
import matplotlib
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        intersections_list = np.ndarray([10])


    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution

        diffrence_func = self.intersect_function(f1, f2)
        self.newton_raphson(a, b, diffrence_func, 0.001)

        X=[0]
        return X

    def intersect_function(self, f1:callable, f2:callable):
        return lambda x: f1(x) - f2(x)


    def newton_raphson(self, left_bracket, right_bracket, function, maxerr):
        h = 0.00000000001
        x = left_bracket
        while x <= right_bracket:
            deriv = (function(x + h) / h) - (function(x) / h)
            epsilon = function(x) / deriv
            lst = []
            while abs(epsilon) >= maxerr:
                epsilon = function(x) / deriv
                # x(i+1) = x(i) - f(x) / f'(x)
                x = x - epsilon
                deriv = (function(x + h) / h) - (function(x) / h)

            print("The value of the root is : ", "%.4f" % x)
            # self.intersections_list.insert(x)
            lst.append(x)
            print(lst)
            x += maxerr


##########################################################################
# # Ploting For Testing use TODO: remove before flight
# if __name__ == "__main__":
#     a = 1
#     b = 12346
#     n = 1000
#
#     def f(x):
#         res = math.sin(x)
#         return res
#
#     intersactor = Assignments2()
#     path = intersector.intersections(f1, f2, a, b)
#     points_x = np.linspace(a, b, num=n)
#     points_y = np.zeros(n)
#     points_y_real = np.zeros(n)
#     for i in range(0, n-1):
#         points_y[i] = path(points_x[i])
#     for i in range(0, n-1):
#         points_y_real[i] = f(points_x[i])
#
#     # extract x & y coordinates of points
#     # x, y = points_values[:, 0], points_values[:, 1]
#     # px, py = path_points[:, 0], path_points[:, 1]
#
#     # plot
#     plt.plot(points_x, points_y)
#     plt.xlim(left=0, right=100)
#     # plt.plot(points_x, points_y_real)
#     plt.show()
#
#     # plt.figure(figsize=(11, 8))
#     # plt.plot(px, py, 'b-')
#     # plt.show()


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_3(self):
        f1 = lambda x: x**2
        f2 = lambda x: 2*x

        a = 0
        b = 2

        intersector = Assignment2()
        intersector.intersections(f1, f2, a, b)

if __name__ == "__main__":
    unittest.main()
