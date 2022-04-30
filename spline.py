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

    def cubic_spline(self, show_plot=False):
        """
        Natural cubic spline.
        :return: zero rate curve and instantaneous forward rate curve
        """
        tck = interpolate.CubicSpline(self.t, self.zero_rate, bc_type='natural')
        xx = np.linspace(0, max(self.t), 600)
        if show_plot:
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


    def Bspline(self, knots=np.linspace(-10, 30, 15), degree=3, show_plot=False):
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

        if show_plot:
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

    def cubic_stability_ratio(self):

        import numpy as np

        # curves before shift
        f_before, z_before = self.cubic_spline()

        # curves after shift
        f_ratios = []
        z_ratios = []
        d_input = 1 / 10000  # parallel shift of 1 bp
        for i in range(len(self.t)):
            zero_rates_i = self.zero_rate
            zero_rates_i[i] += d_input
            PCF_after = Spline_fitting(self.t, zero_rates_i)
            f_after, z_after = PCF_after.cubic_spline()

            # compute stability ratios
            f_ratios.append(max(f_after - f_before) / d_input)
            z_ratios.append(max(z_after - z_before) / d_input)

        # compute mean of the ratios
        f_ratio = round(np.mean(np.array(f_ratios)), 4)
        z_ratio = round(np.mean(np.array(z_ratios)), 4)

        return f_ratio, z_ratio

    def Bspline_stability_ratio(self):

        import numpy as np

        # curves before shift
        f_before, z_before = self.Bspline()

        # curves after shift
        f_ratios = []
        z_ratios = []
        d_input = 1 / 10000  # parallel shift of 1 bp
        for i in range(len(self.t)):
            zero_rates_i = self.zero_rate
            zero_rates_i[i] += d_input
            PCF_after = Spline_fitting(self.t, zero_rates_i)
            f_after, z_after = PCF_after.Bspline()

            # compute stability ratios
            f_ratios.append(max(f_after - f_before) / d_input)
            z_ratios.append(max(z_after - z_before) / d_input)

        # compute mean of the ratios
        f_ratio = round(np.mean(np.array(f_ratios)), 4)
        z_ratio = round(np.mean(np.array(z_ratios)), 4)

        return f_ratio, z_ratio

    def cubic_spline_forward(self, s, t, d=0.5):
        """Computes the forward rate between time s and t (as valued at time 0) using cubic spline interpolation.
        @param s: time to where the forward rate is used to discount to (s < t)
        @param t: time from which the forward rate is used to discount from
        @param d: day count factor for the period [s,t]
        """

        import numpy as np
        import scipy.integrate as integrate

        # compute instantaneous forward curve
        S = Spline_fitting(self.t, self.zero_rate)
        zero_curve, f_curve = S.cubic_spline()

        # discretize the time space and forward curve between time s and t
        t_vec = np.linspace(s, t, 500)
        f_vec = self.get_interpolation(t_vec, f_curve)

        # integrate the forward curve between s and t
        integral = integrate.simpson(f_vec, t_vec)

        # compute the forward rate
        forward_rate = (1 / d) * (np.exp(integral) - 1)

        return forward_rate

    def Bspline_forward(self, s, t, d=0.5):
        """Computes the forward rate between time s and t (as valued at time 0) using B-spline interpolation.
        @param s: time to where the forward rate is used to discount to (s < t)
        @param t: time from which the forward rate is used to discount from
        @param d: day count factor for the period [s,t]
        """

        import numpy as np
        import scipy.integrate as integrate

        # compute instantaneous forward curve
        S = Spline_fitting(self.t, self.zero_rate)
        zero_curve, f_curve = S.Bspline()

        # discretize the time space and forward curve between time s and t
        t_vec = np.linspace(s, t, 500)
        f_vec = self.get_interpolation(t_vec, f_curve)

        # integrate the forward curve between s and t
        integral = integrate.simpson(f_vec, t_vec)

        # compute the forward rate
        forward_rate = (1 / d) * (np.exp(integral) - 1)

        return forward_rate


    def cubic_spline_swap_price(self, T, d=0.5, a=0.5):
        """Computes the break-even fixed/float swap price given an array of forward rates and discount rates for each
        payment. Default is semi-annual payments for both fixed and floating legs."""

        import numpy as np

        # payment periods for float and fixed leg
        t_vec_float = np.arange(0, T + d, d)
        t_vec_fixed = np.arange(0, T + a, a)

        # compute forward rates and discount factors for the float payments
        F_rates = np.zeros(len(t_vec_float) - 1)
        for i in range(1, len(t_vec_float)):
            F_rates[i-1] = self.cubic_spline_forward(t_vec_float[i-1], t_vec_float[i], d)

        # compute the discount factors for the fixed and float payments
        zero_curve = self.cubic_spline()[0]
        r_fixed = self.get_interpolation(t_vec_fixed[1:], zero_curve)
        r_float = self.get_interpolation(t_vec_float[1:], zero_curve)
        df_fixed = np.exp(-(r_fixed * t_vec_fixed[1:]))
        df_float = np.exp(-(r_float * t_vec_float[1:]))

        # compute breakeven swap price
        pv_float = np.sum(d * F_rates * df_float)  # PV of the floating leg
        annuity = np.sum(a * df_fixed)
        swap_price = pv_float / annuity

        return swap_price

    def Bspline_swap_price(self, T, d=0.5, a=0.5):
        """Computes the break-even fixed/float swap price given an array of forward rates and discount rates for each
        payment. Default is semi-annual payments for both fixed and floating legs."""

        import numpy as np

        # payment periods for float and fixed leg
        t_vec_float = np.arange(0, T + d, d)
        t_vec_fixed = np.arange(0, T + a, a)

        # compute forward rates and discount factors for the float payments
        F_rates = np.zeros(len(t_vec_float) - 1)
        for i in range(1, len(t_vec_float)):
            F_rates[i-1] = self.Bspline_forward(t_vec_float[i-1], t_vec_float[i], d)

        # compute the discount factors for the fixed and float payments
        zero_curve = self.Bspline()[0]
        r_fixed = self.get_interpolation(t_vec_fixed[1:], zero_curve)
        r_float = self.get_interpolation(t_vec_float[1:], zero_curve)
        df_fixed = np.exp(-(r_fixed * t_vec_fixed[1:]))
        df_float = np.exp(-(r_float * t_vec_float[1:]))

        # compute breakeven swap price
        pv_float = np.sum(d * F_rates * df_float)  # PV of the floating leg
        annuity = np.sum(a * df_fixed)
        swap_price = pv_float / annuity

        return swap_price


if __name__ == "__main__":

    data = pd.read_csv('data/zero_rate.csv', index_col=0)/100
    data.sort_index(inplace=True)
    # print(data)
    tenor = [0.5,1,2,3,4,5,6,7,8,9,10,15,20,30]

    # create a Spline_fitting object
    S = Spline_fitting(tenor, list(data.loc['2020-03']))

    # output: zero rate grid and instantaneous forward rate grid
    zero_rate1, f_rate1 = S.cubic_spline(show_plot=True)
    zero_rate2, f_rate2 = S.Bspline(show_plot=True)

    # get interpolation of some time spots we want
    tt = 1.5  # the given maturity
    print(S.get_interpolation(tt, f_rate1))  # get zero rate for this point








