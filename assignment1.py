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
        c = np.ones(n)
        b = np.full(n+1, 4)
        b[0] = 2
        b[n] = 7
        a = np.ones(n)
        a[n-1] = 2

        a_c_points = self.thomas(a, b, c, P)
        # solve system, find a & b
        # a_c_points = np.linalg.solve(C, P)  # TODO: change to thomas algo!
        B = [0] * n
        for i in range(n - 1):
            B[i] = 2 * points[i + 1] - a_c_points[i + 1]
        B[n - 1] = (a_c_points[n - 1] + points[n]) / 2

        return a_c_points, B

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

    def final_poly(self, points):
        return self.get_bezier_cubic(points)

    def x_location(self, value_of_x, points):
        start_point = 0
        end_point = 0

        for key, val in self.interpolated_dict.items():
            if start_point == 0:
                start_point = key
            if value_of_x > key[0]:
                start_point = key
            else:
                end_point = key
                break

        if value_of_x == start_point[0] or end_point[0] == start_point[0]:
            normal_x = 0
        else:
            normal_x = (value_of_x - start_point[0])/(end_point[0] - start_point[0])

        return self.interpolated_dict[start_point](normal_x)[1]

    def TDMAsolver(self, lower, middle, upper, points_vector):
        equations_num = len(lower)
        cLower, cMiddle, cUpper, cVector = map(np.array, (lower, middle, upper, points_vector))
        for i in range(1, equations_num):
            mc = cLower[i] / cMiddle[i - 1]
            cMiddle[i] = cMiddle[i] - mc * cUpper[i - 1]
            cVector[i][0] = cVector[i][0] - mc * cVector[i - 1][0]
            cVector[i][1] = cVector[i][1] - mc * cVector[i - 1][1]

        xc = np.empty([equations_num, 2])
        for i in range(0, len(cLower)):
            xc[i][0] = cLower[i]
        xc[-1] = cVector[-1] / cMiddle[-1]
        # xc[-1][1] = cVector[-1][1] / cMiddle[-1]

        for j in range(equations_num - 2, -1, -1):
            xc[j][0] = (cVector[j][0] - cUpper[j] * xc[j + 1][0]) / cMiddle[j]
            xc[j][1] = (cVector[j][1] - cUpper[j] * xc[j + 1][0]) / cMiddle[j]

        del cMiddle, cUpper, cVector  # delete variables from memory
        return xc

    def thomas(self, lower, middle, upper, points_vector):
        equations_num = len(lower)
        for i in range(0, equations_num):
            dx = points_vector[i][0]
            dy = points_vector[i][1]
            prev_d_x = points_vector[i-1][0]
            prev_d_y = points_vector[i-1][1]

            m = lower[i-1]/middle[i-1]
            middle[i] = middle[i] - m * upper[i-1]
            dx2 = dx - m * prev_d_x
            dy2 = dy - m * prev_d_y

            points_vector[i] = [dx2, dy2]

        last_cpx = points_vector[-1][0] / middle[-1]
        last_cpy = points_vector[-1][1] / middle[-1]
        x = points_vector
        x[-1] = [last_cpx, last_cpy]

        for i in range(equations_num-2, -1, -1):
            cp_x = (points_vector[i][0] - upper[i] * x[i+1][0]) / middle[i]
            cp_y = (points_vector[i][1] - upper[i] * x[i+1][1]) / middle[i]
            x[i] = [cp_x, cp_y]
        return x


# Ploting For Testing use TODO: remove before flight
# if __name__ == "__main__":
#     a = -100
#     b = 100
#     n = 100
#
#     # def f(x):
#     #     res = math.sin(x)
#     #     return res
#     a_f = np.random.randn(2)
#     f = np.poly1d(a_f)
#
#     interpolator = Assignment1()
#     path = interpolator.interpolate(f, a, b, n)
#     points_x = np.linspace(a, b, num=n-1)
#     points_y = np.zeros(n)
#     points_y_real = np.zeros(n)
#     for i in range(0, n):
#         points_y[i] = path(points_x[i])
#     for i in range(0, n):
#         points_y_real[i] = f(points_x[i])
#
#     # extract x & y coordinates of points
#     # x, y = points_values[:, 0], points_values[:, 1]
#     # px, py = path_points[:, 0], path_points[:, 1]
#
#     # plot
#     plt.plot(points_x, points_y, 'r')
#     plt.xlim(left=0, right=1000)
#     plt.plot(points_x, points_y_real, 'b')
#     plt.show()
#
#     # plt.figure(figsize=(11, 8))
#     # plt.plot(px, py, 'b-')
#     # plt.show()


#########################################################################


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

    # def test_3(self):
if __name__ == "__main__":
    unittest.main()
