"""
In this assignment you should interpolate the given function.
"""



import numpy as np
import time
import random
import matplotlib.pyplot as plt
import math


class Assignment1:
    # TODO: Change functions name and order!!!!!
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        self.interpolated_dict = None

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        # Get n random numbers (floats) in range of [a,b]
        points_x_values = np.linspace(a, b, num=n)
        points_values = np.zeros([n, 2])
        for i in range(0, n):
            _x = points_x_values.item(i)
            points_values[i] = [_x, f(_x)]
        self.interpolated_dict = self.final_poly(points_values)
        return lambda x: self.x_location(x, points_values)

    # find the a & b points

    def get_bezier_coef(self, points):
        # since the formulas work given that we have n+1 points
        # then n must be this:
        n = len(points) - 1  # TODO: give "n" as a parameter to the function, replace the initialization

        # build coefficients matrix
        # C = 4 * np.identity(n)
        # np.fill_diagonal(C[1:], 1)
        # np.fill_diagonal(C[:, 1:], 1)
        # C[0, 0] = 2
        # C[n - 1, n - 1] = 7
        # C[n - 1, n - 2] = 2

        # build points vector
        P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
        P[0] = points[0] + 2 * points[1]
        P[n - 1] = 8 * points[n - 1] + points[n]

        # creating arrays for coefficients
        c = np.ones(n-1)
        b = np.full(n-1, 4)
        b[0] = 2
        b[n-1] = 7
        a = np.full(n-1, 1)
        a[n-1] = 2

        A = self.TDMAsolver(a, b, c, P)
        # solve system, find a & b
        # A = np.linalg.solve(C, P)  # TODO: change to thomas algo!
        B = [0] * n
        for i in range(n - 1):
            B[i] = 2 * points[i + 1] - A[i + 1]
        B[n - 1] = (A[n - 1] + points[n]) / 2

        return A, B

    # returns the general Bezier cubic formula given 4 control points
    def get_cubic(self, a, b, c, d):
        return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * \
                         np.power(t, 2) * c + np.power(t, 3) * d

    # return one cubic curve for each consecutive points
    def get_bezier_cubic(self, points):
        A, B = self.get_bezier_coef(points)
        return {(points[i][0], points[i][1]):
                self.get_cubic(points[i], A[i], B[i], points[i + 1]) for i in range(len(points) - 1)
                }

    def final_poly(self, points): # TODO: n is needed?
        return self.get_bezier_cubic(points)

    def x_location(self, value_of_x, points):
        start_point = 0
        end_point = 0

        for key, val in self.interpolated_dict.items():  # TODO: Check End Point
            if start_point == 0:
                start_point = key
            if value_of_x > key[0]:
                start_point = key
            else:
                end_point = key

        normal_x = (value_of_x - start_point[0])/(end_point[0] - start_point[0])

        return self.interpolated_dict[start_point](normal_x)[1]

    def TDMAsolver(self, a, b, c, d):
        nf = len(a)  # number of equations
        ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy the array
        for it in range(1, nf):
            mc = ac[it] / bc[it - 1]
            bc[it] = bc[it] - mc * cc[it - 1]
            dc[it] = dc[it] - mc * dc[it - 1]

        xc = ac
        xc[-1] = dc[-1] / bc[-1]

        for il in range(nf - 2, -1, -1):
            xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

        del bc, cc, dc  # delete variables from memory
        return xc


# if __name__ == "__main__":
#     a = 1
#     b = 12346
#     n = 100
#
#     def f(x):
#         res = math.cos(x)
#         return res
#
#     # points_x_values = np.linspace(a, b, num=n + 1)
#     # points_values = np.zeros([n+1, 2])
#     # for i in range(0, n+1):
#     #     _x = points_x_values.item(i)
#     #     points_values[i] = [_x, f(_x)]
#     #
#     interpolator = Assignment1()
#     path = interpolator.interpolate(f, a, b, n)
#
#     # extract x & y coordinates of points
#     # x, y = points_values[:, 0], points_values[:, 1]
#     px, py = path[:, 0], path[:, 1]
#
#     # plot
#     plt.plot(path[:, 0], path[:, 1])
#     plt.show()
#     plt.figure(figsize=(11, 8))
#     plt.plot(px, py, 'b-')
#     # plt.plot(x, y, 'ro')
#     plt.show()






















##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
