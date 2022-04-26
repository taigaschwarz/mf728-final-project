"""
Program:
Author: cai
Date: 2022-04-24
"""

# import math
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 3000)


class Spline_fitting:

    def __init__(self, t, zero_rate):

        self.t = t
        self.zero_rate = zero_rate


    # def quadratic_spline(self):
    #     self.t.insert(0, 0)
    #     self.zero_rate.insert(0, 0)
    #     tck = interpolate.interp1d(self.t, self.zero_rate, kind='quadratic')
    #     xx = np.linspace(min(self.t), max(self.t), 600)
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(self.t, self.zero_rate, 'o', xx, tck(xx))
    #     plt.xlabel('tenor')
    #     plt.ylabel('zero rates')
    #     plt.title('zero curve (quadratic splines)')
    #     plt.grid(True)
    #     plt.show()
    #     return tck(xx)

    def cubic_spline(self):
        """
        :return: zero rate curve and instantaneous forward rate curve
        """
        tck = interpolate.CubicSpline(self.t, self.zero_rate, bc_type='natural')
        xx = np.linspace(0, max(self.t), 600)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.t, self.zero_rate, 'o', xx, tck(xx))
        plt.xlabel('tenor')
        plt.ylabel('zero rates')
        plt.title('zero curve (cubic splines)')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(xx, tck(xx, 1) * xx + tck(xx))
        plt.xlabel('time')
        plt.ylabel('f rates')
        plt.title('instantaneous forward curve (cubic splines)')
        plt.grid(True)
        plt.show()
        return tck(xx), tck(xx, 1) * xx + tck(xx)


    def Bspline(self, knots=np.linspace(-10, 30, 15), degree=3):
        """
        :return: zero rate curve and instantaneous forward rate curve
        """
        def least_square(knots, degree, coefficients):
            # coefficients = [c1,c2,c3,c4]
            # make the square of difference between market rate and model rate least
            spl = interpolate.BSpline(knots, coefficients, degree)
            re = 0
            for i in range(len(self.t)):
                re += 0.5 * (spl.__call__(self.t[i]) - self.zero_rate[i]) ** 2
            # mu = np.linspace(3, 6, len(self.t)-1)
            # adding convexity regularization
            for i in range(len(self.t)-1):
                temp = ((spl.__call__(self.t[i], nu=2) ** 2 + spl.__call__(self.t[i+1], nu=2) ** 2) * (self.t[i+1] - self.t[i])) * 0.5
                re += 0.5 * 2 * temp
            return re * 100
        # optimizaion
        func = lambda x: (least_square(knots, degree, x))
        opti = minimize(func, np.zeros(len(knots)-degree-1), method='SLSQP')
        result = opti.x

        spl = interpolate.BSpline(knots, result, degree)
        xx = np.linspace(0, max(self.t), 600)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.t, self.zero_rate, 'o', xx, spl(xx))
        plt.xlabel('tenor')
        plt.ylabel('zero rates')
        plt.title('zero curve (B-splines)')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(xx, spl(xx, 1) * xx + spl(xx))
        plt.xlabel('time')
        plt.ylabel('f rates')
        plt.title('instantaneous forward curve (B-splines)')
        plt.grid(True)
        plt.show()
        return spl(xx), spl(xx, 1) * xx + spl(xx)

    def get_interpolation(self, points, rate_grid):
        re = np.interp(points, np.linspace(0, max(self.t), 600), rate_grid)
        return re


if __name__ == "__main__":

    data = pd.read_csv('/Users/cai/python_program/MF728/data/zero_rate.csv', index_col=0)
    data.sort_index(inplace=True)
    # print(data)
    tenor = [0.5,1,2,3,4,5,6,7,8,9,10,15,20,30]

    # create a Spline_fitting object
    S = Spline_fitting(tenor, list(data.loc['2017-04']))

    # output: zero rate grid and instantaneous forward rate grid
    zero_rate1, f_rate1 = S.cubic_spline()
    zero_rate2, f_rate2 = S.Bspline()

    # get interpolation of some time spots we want
    tt = 1.5  # the given maturity
    print(S.get_interpolation(tt, zero_rate1))  # get zero rate for this point



    # fig, ax = plt.subplots()
    # for i in data.columns:
    #     S = Spline_fitting(tenor, data[i])
    #     zero, forward = S.cubic_spline()
    #     ax.plot(S.t, S.zero_rate, 'o', np.linspace(0, max(S.t), 600), zero)
    #     # ax.text(1, 1, 'Time={}s'.format(i / 100))
    #     plt.pause(1e-2)
    #     ax.cla()
    # plt.show()







