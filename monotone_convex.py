"""
Program:
Author: cai
Date: 2022-04-26
"""
import numpy as np
import pandas as pd
from scipy.integrate import quad
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
        xx = np.linspace(0, max(self.t), 600)
        func = lambda x: self.construct_g_func(x)
        zero_curve = []
        instantaneous_curve = []
        for i in xx:
            zero_curve.append(quad(func, 0, i)[0] / i)
            instantaneous_curve.append(self.construct_g_func(i))
        return zero_curve, instantaneous_curve

if __name__ == "__main__":

    data = pd.read_csv('data/zero_rate.csv', index_col=0)
    data.sort_index(inplace=True)
    zero_rate = list(data.loc['2022-03'])
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



    


