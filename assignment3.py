"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random
import assignment2


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        self.intersector = assignment2.Assignment2()

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        # replace this line with your solution
        integral = self.simpsons_rule(f, a, b, n)
        result = np.float32(integral)
        return result

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        # replace this line with your solution
        intersaction_function = self.intersect_function(f1, f2)
        abs_intersaction_function = self.abs_intersect_function(f1, f2)
        intersections = self.find_intersections(f1, f2)
        intersections_no = self.count_iterable(intersections)

        area = 0

        area += self.integrate(abs_intersaction_function, 1, intersections[0], 100)
        for i in range(0, intersections_no-1):
            area += self.integrate(abs_intersaction_function, intersections[i], intersections[i+1], 100)
        area += self.integrate(abs_intersaction_function, intersections[intersections_no-1], 100, 100)

        result = np.float32(area)
        return result

    def simpsons_rule(self, function, left_bracket, right_bracket, n):

        # Calculating the value of h
        h = (right_bracket - left_bracket) / n

        # List for storing value of x and f(x)
        x_values = list()
        function_values = list()

        # Calculating values of x and f(x)
        i = 0
        while i <= n:
            x_values.append(left_bracket + i * h)
            function_values.append(function(x_values[i]))
            i += 1

        # Calculating result
        result = 0
        i = 0
        while i <= n:
            if i == 0 or i == n:
                result += function_values[i]
            elif i % 2 != 0:
                result += 4 * function_values[i]
            else:
                result += 2 * function_values[i]
            i += 1
        result = result * (h / 3)
        return result

    def intersect_function(self, f1: callable, f2: callable):
        return lambda x: f1(x) - f2(x)

    def abs_intersect_function(self, f1: callable, f2: callable):
        return lambda x: abs(f1(x) - f2(x))

    def find_intersections(self, f1: callable, f2: callable):
        intersections = self.intersector.intersections(f1, f2, 1, 100)
        return intersections

    def count_iterable(self, iterable):
        return sum(1 for e in iterable)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_3(self):
        ass3 = Assignment3()
        f1 = lambda x: x*np.sin(x)

        res = ass3.integrate(f1, 0, np.pi/2, 20)
        print(res)

    def test_4(self):
        ass3 = Assignment3()
        f1 = lambda x: np.sin(x)
        f2 = lambda x: np.cos(x)

        print(ass3.areabetween(f1, f2))

    def test_5_stack(self):
        ass3 = Assignment3()
        f1 = lambda x: (x**(np.e-1))*np.e**(-x)
        f2 = lambda x: np.cos(x)

        res = ass3.integrate(f1, 2, 5, 4)
        print(res)




if __name__ == "__main__":
    unittest.main()
