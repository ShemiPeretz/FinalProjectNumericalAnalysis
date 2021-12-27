"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
import matplotlib
import math
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        self.intersections_list = []
        self.count_intersections = 0

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
        guesses = np.linspace(a, b, 5000)
        #guesses = self.filter_points(points, diffrence_func, maxerr)

        for i in range(0, len(guesses)-1):
            if guesses[i] is not None:
                b_guess = self.bisection_guesser(guesses[i], guesses[i+1], diffrence_func, maxerr)
            if b_guess is not None:
                self.newton_raphson(b_guess, a, b, diffrence_func, maxerr)

        print(self.intersections_list)
        print(len(self.intersections_list))
        self.count_intersections = len(self.intersections_list)
        return self.intersections_list

    def intersect_function(self, f1:callable, f2:callable):
        return lambda x: f1(x) - f2(x)

    def filter_points(self, points, function, maxerr):
        filtered = []
        h = 0.00000000001
        deriv = lambda x: ((function(x + h) / h) - (function(x) / h))
        for i in range(len(points)-1):
            if not (points[i + 1] - points[i] < maxerr):
                filtered.append(points[i])
        return filtered

    def bisection_guesser(self, left_bracket, right_bracket, function, maxerr):
        # TODO: What if a and b arent legal? make it work!
        new_err = maxerr*10
        if (function(left_bracket) * function(right_bracket)) > 0:
            return
        if function(left_bracket) == 0:
            return left_bracket
        if function(right_bracket) == 0:
            return right_bracket
        c = left_bracket
        self.max_x = (left_bracket + right_bracket) / 2

        while (right_bracket - left_bracket) >= new_err:

            # Find middle point
            c = (left_bracket + right_bracket) / 2
            if c > self.max_x:
                self.max_x = c

            # Check if middle point is root
            if function(c) == 0.0:
                break

            # Decide the side to repeat the steps
            if function(c) * function(left_bracket) < 0:
                right_bracket = c
            else:
                left_bracket = c

        return c

    def secant_guesser(self, x1, x2, function, maxerr):
        new_err = maxerr*10
        n = 0
        xm = 0
        x0 = 0
        c = 0
        if function(x1) == 0:
            return x1
        if function(x2) == 0:
            return x2
        if function(x1) * function(x2) < 0:
            while True:
                # calculate the intermediate value
                x0 = (x1 * function(x2) - x2 * function(x1)) / (function(x2) - function(x1))

                # check if x0 is root of
                # equation or not
                c = function(x1) * function(x0)

                # update the value of interval
                x1 = x2
                x2 = x0

                # update number of iteration
                n += 1

                # if x0 is the root of equation
                # then break the loop
                if c == 0:
                    break
                xm = ((x1 * function(x2) - x2 * function(x1)) / (function(x2) - function(x1)))

                if abs(xm - x0) < new_err:
                    break

            return x0

        else:
            # TODO: What do you do in that situation
            return

    def newton_raphson(self, guess, left_bracket, right_bracket, function, maxerr):
        self.flag = False
        h = 0.00000000000001
        x = guess
        deriv = (function(x + h) / h) - (function(x) / h)
        epsilon = function(x) / deriv
        while abs(function(x)) >= maxerr:
            epsilon = function(x) / deriv
            # x(i+1) = x(i) - f(x) / f'(x)
            x = x - epsilon
            deriv = (function(x + h) / h) - (function(x) / h)

        # print("The value of the root is : ", "%.4f" % x)
        if x in self.intersections_list:
            return
        else:
            self.intersections_list.append(x)


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
        f3 =lambda x: 1.52*x**10 - 0.6189*x**9 + 1.58*x**8 - 0.01277*x**7 - 0.6578*x**6 + 0.3955*x**5 + 1.223*x**4 + \
                      1.716*x**3 - 0.4981*x**2 + 0.3833*x - 1.359
        f4 =lambda x: -0.2656*x**10+0.1583*x**9-0.1591*x**8+0.9435*x**7+1.406*x**6-0.4601*x**5-0.1547*x**4-0.5351*x**3 - \
                      1.107*x**2+0.102*x +1.763
        f1 = lambda x: x**2
        f2 = lambda x: 2*x

        f5 = lambda x: math.sin(100*x)
        f6 = lambda x: math.cos(100*x)
        f7 = lambda x: (x*np.log10(x))/12
        f8 = lambda x: 0
        f9 = lambda x: 10 * x * np.sin(x)

        a = -10
        b = 10

        intersector = Assignment2()
        intersector.intersections(f5, f6, a, b)

    def test4_ass2_2(self):

        def zero(a): return 0

        def sin(x): return 10 * x * np.sin(x)

        f1 = lambda x: (x*math.log10(x))/12

        domain = (0, 5)
        ass = Assignment2()
        roots = ass.intersections(f1, zero, domain[0], domain[1], maxerr=0.001)

        for i in roots:
            self.assertLess(abs(sin(i)), 0.001, msg="Err bigger the 0.001")

if __name__ == "__main__":
    unittest.main()
