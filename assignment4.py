"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        # replace these lines with your solution

        n = 10000

        x_values = np.linspace(a, b, n)
        y_values = np.zeros(n)
        for i in range(n):
            y_values[i] = f(x_values[i])

        x_sum = x_values.sum()
        y_sum = y_values.sum()

        self.least_squares(x_values, y_values, x_sum, y_sum, d, n+0.0)






        result = lambda x: x
        y = f(1)

        return result

    def least_squares(self, x_values, y_values,xSum, ySum, d, n):

        sum_xy = 0
        sum_x_sqr = 0
        for i in range(int(n)):
            sum_x_sqr += x_values[i] ** 2
        for i in range(int(n)):
            sum_xy += (x_values[i] * y_values[i])
        # coef = np.zeros(d)
        # for i in range(1, d):
        #

        A = np.array([[sum_x_sqr, xSum], [xSum, n]])
        B = np.array([sum_xy, ySum])

        AT = A.T
        ATA = np.dot(A.T, A)
        ATA_inv = np.linalg.inv(ATA)
        ATA_invAT = np.dot(ATA_inv, AT)

        coef = np.dot(ATA_invAT, B)


        # coef = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), B)

        print(coef)

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):            
            self.assertNotEquals(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)


    def test_4(self):
        ass4 = Assignment4A()
        f = poly(2, 0)
        nf = NOISY(1)(f)
        print(f)
        print(nf(5))

        ass4.fit(nf, 0, 10, 1, maxtime=10)

        



if __name__ == "__main__":
    unittest.main()
