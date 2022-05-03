"""
Program:
Author: cai
Date: 2022-04-26
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class Monotone_convex:

    def __init__(self, t, zero_rate):
        self.t = t
        self.zero_rate = zero_rate

    def find_discrete_forward_rates(self):
        re = []
        for i in range(len(self.t)):
            if i == 0:
                re.append(self.zero_rate[i])
            else:
                re.append((self.zero_rate[i]*self.t[i]-self.zero_rate[i-1]*self.t[i-1])/ (self.t[i]-self.t[i-1]))
        return re

    def find_f_rate_nodes(self):
        fd = self.find_discrete_forward_rates()
        re = []
        for i in range(len(fd)-1):
            if i == 0:
                temp = (self.t[i] - 0) / (self.t[i + 1] - 0) * fd[i + 1]
                temp2 = (self.t[i + 1] - self.t[i]) / (self.t[i + 1] - 0) * fd[i]
                re.append(temp + temp2)
            else:
                temp = (self.t[i]-self.t[i-1])/(self.t[i+1]-self.t[i-1]) * fd[i+1]
                temp2 = (self.t[i+1]-self.t[i])/(self.t[i+1]-self.t[i-1]) * fd[i]
                re.append(temp+temp2)

        # f is required to be everywhere positive
        temp3 = fd[0] - 0.5 * (re[0] - fd[0])
        re.insert(0, max(0, min(temp3, 2*fd[0])))
        temp4 = fd[-1] - 0.5 * (re[-1] - fd[-1])
        re.insert(-1, max(0, min(temp4, 2*fd[-1])))
        return re

    def construct_g_func(self, time):
        global g0, g1, x, idx
        fd_grid = self.find_discrete_forward_rates()
        f_grid = self.find_f_rate_nodes()
        time_grid = np.append([0], self.t)
        if time <= 0:
            return f_grid[0]
        elif time >= time_grid[-1]:
            return f_grid[-1]
        else:
            for i in range(len(time_grid)):
                if time_grid[i] < time <= time_grid[i+1]:
                    x = (time - time_grid[i]) / (time_grid[i+1] - time_grid[i])
                    g0 = f_grid[i] - fd_grid[i]
                    g1 = f_grid[i+1] - fd_grid[i]
                    idx = i
                else:
                    continue
            if x == 0:
                G = g0
            elif x == 1:
                G = g1
            elif (g0 < 0 and -0.5 * g0 <= g1 and g1 <= -2 * g0) or (g0 > 0 and -0.5 * g0 >= g1 and g1 >= -2 * g0):
                G = g0 * (1 - 4 * x + 3 * x ** 2) + g1 * (-2 * x + 3 * x ** 2)
            elif (g0 < 0 and g1 > -2 * g0) or (g0 > 0 and g1 < -2 * g0):
                eta = (g1 + 2 * g0) / (g1 - g0)
                if x <= eta:
                    G = g0
                else:
                    G = g0 + (g1 - g0) * ((x - eta) / (1 - eta)) ** 2
            elif (g0 > 0 and 0 > g1 and g1 > -0.5 * g0) or (g0 < 0 and 0 < g1 and g1 < -0.5 * g0):
                eta = 3 * g1 / (g1 - g0)
                if x <= eta:
                    G = g1 + (g0 - g1) * ((eta - x) / eta) ** 2
                else:
                    G = g1
            elif g0 == 0 and g1 == 0:
                G = 0
            else:
                eta = g1 / (g0 + g1)
                A = -g0 * g1 / (g0 + g1)
                if x < eta:
                    G = A + (g0 - A) * ((eta - x) / eta) ** 2
                else:
                    G = A + (g1 - A) * ((eta - x) / (1 - eta)) ** 2
        return G + fd_grid[idx]

    def fitting(self):
        """
        :return: zero curve and instantaneous f curve
        """

        from scipy.integrate import quad

        xx = np.linspace(0, max(self.t), 600)
        func = lambda x: self.construct_g_func(x)
        zero_curve = []
        instantaneous_curve = []
        for i in xx:
            zero_curve.append(quad(func, 0, i)[0] / i)
            instantaneous_curve.append(self.construct_g_func(i))
        return np.array(zero_curve), np.array(instantaneous_curve)

    def stability_ratio(self):

        import numpy as np

        # curves before shift
        f_before, z_before = self.fitting()

        # curves after shift
        f_ratios = []
        z_ratios = []
        d_input = 1 / 10000  # parallel shift of 1 bp
        for i in range(len(self.t)):
            zero_rates_i = self.zero_rate
            zero_rates_i[i] += d_input
            PCF_after = Monotone_convex(self.t, zero_rates_i)
            f_after, z_after = PCF_after.fitting()

            # compute stability ratios
            f_ratios.append(max(f_after - f_before) / d_input)
            z_ratios.append(max(z_after - z_before) / d_input)

        # compute mean of the ratios
        f_ratio = round(np.mean(np.array(f_ratios)), 4)
        z_ratio = round(np.mean(np.array(z_ratios)), 4)

        return f_ratio, z_ratio

    def get_interpolation(self, points, rate_grid):

        import numpy as np
        re = np.interp(points, np.linspace(0, max(self.t), 600), rate_grid)
        return re

    def monotone_convex_forward(self, s, t, d=0.5):
        """Computes the forward rate between time s and t (as valued at time 0) using monotone convex interpolation.
        @param s: time to where the forward rate is used to discount to (s < t)
        @param t: time from which the forward rate is used to discount from
        @param d: day count factor for the period [s,t]
        """

        import numpy as np
        import scipy.integrate as integrate

        # compute instantaneous forward curve
        zero_curve, f_curve = self.fitting()

        # discretize the time space and forward curve between time s and t
        t_vec = np.linspace(s, t, 500)
        f_vec = self.get_interpolation(t_vec, f_curve)

        # integrate the forward curve between s and t
        integral = integrate.simpson(f_vec, t_vec)

        # compute the forward rate
        forward_rate = (1 / d) * (np.exp(integral) - 1)

        return forward_rate

    def mc_swap_price(self, T, d=0.5, a=0.5):
        """Computes the break-even fixed/float swap price given an array of forward rates and discount rates for each
        payment. Default is semi-annual payments for both fixed and floating legs."""

        import numpy as np

        # payment periods for float and fixed leg
        t_vec_float = np.arange(0, T + d, d)
        t_vec_fixed = np.arange(0, T + a, a)

        # compute forward rates and discount factors for the float payments
        F_rates = np.zeros(len(t_vec_float) - 1)
        for i in range(1, len(t_vec_float)):
            F_rates[i-1] = self.monotone_convex_forward(t_vec_float[i-1], t_vec_float[i], d)

        # compute the discount factors for the fixed and float payments
        zero_curve = self.fitting()[0]
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

    data = pd.read_csv('data/zero_rate.csv', index_col=0)
    data.sort_index(inplace=True)
    zero_rate = list(data.loc['2020-10'])
    tenor = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]

    M = Monotone_convex(tenor, zero_rate)
    zero_curve, inst_f_curve = M.fitting()

    plt.subplot(1, 2, 1)
    plt.plot(tenor, zero_rate, 'o', np.linspace(0, max(tenor), 600), zero_curve)
    plt.xlabel('tenor')
    plt.ylabel('zero rates')
    plt.title('zero curve (monotone convex)')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0, max(tenor), 600), inst_f_curve)
    plt.xlabel('time')
    plt.ylabel('f rates')
    plt.title('instantaneous forward curve (monotone convex)')
    plt.grid(True)
    plt.show()



    


