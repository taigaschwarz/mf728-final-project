# author: Taiga Schwarz
# date modified: 04/22/2022
# Raw Interpolation Method (piecewise constant forward rate)

# helper functions
def swap_to_CFR(a, d, s, F_prev, prev_intervals, cur_interval):
    """ We assume semi-annual payment structure;
       F_prev = LIBOR/OIS forward rates of previous term periods;
                can be 0 or a numpy array of the fixed constant forward rates of previous terms
       d = daycount values of the Nth term period applying to the floating leg
       a = daycount values of the Nth term period applying to the fixed leg
       intervals = array of time intervals between term maturities of input swaps """

    import sympy
    import numpy as np

    F = sympy.Symbol('F')  # constant LIBOR forward rate

    if type(F_prev).__module__ == np.__name__ and len(F_prev) >= 1:
        numerator1 = 2 * prev_intervals * F_prev * (F_prev * d + 1) ** (-1)  # previous terms
        numerator2 = 2 * cur_interval * F * (F * d + 1) ** (-1)  # current term
        numerator = np.sum(numerator1) + numerator2
        denominator1 = 2 * prev_intervals * (F_prev * a + 1) ** (-1)  # previous terms
        denominator2 = 2 * cur_interval * (F * a + 1) ** (-1)  # current term
        denominator = np.sum(denominator1) + denominator2
        F = sympy.solve(numerator / denominator - s, F)[0]

    elif F_prev == 0:
        F = s

    return F


def forward_to_df(forward_rates, intervals, m):
    """Derives the discount factors given corresponding forward rates.

    :param forward_rates: array of forward rates
    :param intervals: array of intervals between market term periods
    :param m: number of payment periods per annum"""

    import numpy as np
    import math

    periods = intervals * m  # array of m payments per interval
    N = np.sum(intervals * m)  # total number of discount factors to solve for
    discount_factors = np.zeros(N)
    f_rates = np.repeat(forward_rates, periods)
    f_sum = 0  # initialize sum of forward rates

    for i in range(len(f_rates)):
        discount_factors[i] = math.exp(-(f_sum + f_rates[i] / 2))
        f_sum += f_rates[i] / 2

    return discount_factors


def df_to_zero(discount_factors, tenors):
    """computes constant zero rates given an array of discount factors"""

    import numpy as np

    r = -(np.log(discount_factors) / tenors)  # zero rates

    return r

# piecewise-constant forward rate swap rate bootstrapping
def piecewise_const_fwd(swap_rates, terms, a, d):
    """Extracts a zero curve by bootstrapping market fixed/float par swap rates and using piecewise
    constant forward rates.

    :param a: daycount fraction for fixed payments
    :param d: daycount fraction for floating payments
    :param swap_rates: an array of input market par swap rates
    :param terms: an array of terms corresponding to the input swap rates
    """

    import numpy as np

    constant_forwards = []

    # extend curve to time 0
    terms = np.insert(terms, 0, 0)
    swap_rates = np.insert(swap_rates, 0, swap_rates[0])
    intervals = np.diff(terms)  # time intervals between input swap term maturity dates

    # extract the constant forward rate that matches the first market swap rate
    F1 = swap_to_CFR(a=a, d=d, s=swap_rates[0], F_prev=0, prev_intervals=0, cur_interval=intervals[0])
    constant_forwards.append(F1)

    # bootstrap the forward rates that enable us to match the rest of the swaps
    F_prev = np.array(constant_forwards)
    prev_intervals = np.array([intervals[0]])
    for i in range(1, len(intervals)):
        F = swap_to_CFR(a=a, d=d, s=swap_rates[i], F_prev=F_prev, prev_intervals=prev_intervals,
                        cur_interval=intervals[i])
        constant_forwards.append(F)
        F_prev = np.insert(F_prev, len(F_prev), F)
        prev_intervals = np.insert(prev_intervals, len(prev_intervals), intervals[i])

    # derive the discount factors from the constant forward rates
    m = int(1 / a)
    discount_factors = forward_to_df(constant_forwards, intervals, m)

    # derive the zero rates from the discount factors
    tenors = np.arange(terms[0], terms[-1], 0.5)
    zero_rates = df_to_zero(discount_factors, tenors)

    return zero_rates, constant_forwards


class Raw_interpolation:

    def __init__(self, init_terms, init_zero_rates):
        """define the attributes of the raw interpolation method"""
        self.terms = init_terms  # the input terms
        self.zero_rates = init_zero_rates  # the input zero rates

    def raw_interpolation(self):
        """Computes the forward and zero rate curves using raw interpolation method (corresponds to being linear
        on the log of the discount factors, i.e. piecewise constant forwards"""

        import numpy as np
        from scipy.interpolate import interp1d

        # extends curve to 0
        t = np.insert(self.terms, 0, 0)
        r = np.insert(self.zero_rates, 0, self.zero_rates[0])

        # compute discrete forward rates and log of capitalization rates (r*t)
        dforwards = np.zeros(len(t)-1)
        rt = np.zeros(len(t))
        rt[0] = r[0]*t[0]  # fill the first element
        for i in range(1, len(t)):
            f = (r[i]*t[i] - r[i-1]*t[i-1])/(t[i] - t[i-1])
            dforwards[i-1] = f
            rt[i] = r[i]*t[i]
        dforwards = np.insert(dforwards, 0, dforwards[0])
        # linearly interpolate
        rt_func = interp1d(t, rt)
        xx = np.linspace(0, max(t), 600)
        zero_curve = rt_func(xx[1:])/xx[1:]
        zero_curve = np.insert(zero_curve, 0, zero_curve[0])
        return dforwards, zero_curve

    def stability_ratio(self):

        import numpy as np

        # curves before shift
        f_before, z_before = self.raw_interpolation()

        # curves after shift
        f_ratios = []
        z_ratios = []
        d_input = 1 / 10000  # parallel shift of 1 bp
        for i in range(len(self.terms)):
            zero_rates_i = self.zero_rates
            zero_rates_i[i] += d_input
            PCF_after = Raw_interpolation(self.terms, zero_rates_i)
            f_after, z_after = PCF_after.raw_interpolation()

            # compute stability ratios
            f_ratios.append(max(f_after - f_before) / d_input)
            z_ratios.append(max(z_after - z_before) / d_input)

        # compute mean of the ratios
        f_ratio = round(np.mean(np.array(f_ratios)), 4)
        z_ratio = round(np.mean(np.array(z_ratios)), 4)
        return f_ratio, z_ratio



if __name__=='__main__':

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    swap_df = pd.read_csv('data/SOFR_swap.csv')
    sofr_swaps = np.array(swap_df.iloc[1, :])[3:]/100  # starting at the 1 year swap
    # terms = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])
    # sofr_swaps = np.array([2.8438, 3.060, 3.126, 3.144, 3.150, 3.169, 3.210, 3.237])/100
    # terms = np.array([1, 2, 3, 4, 5, 7, 10, 30])

    # zero_rates, constant_forwards = piecewise_const_fwd(sofr_swaps, terms, 0.5, 0.5)
    #
    # tenors = np.arange(0, 30, 0.5)
    #
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(tenors, zero_rates, 'orange')
    # plt.xlabel('tenor')
    # plt.ylabel('zero rates')
    # plt.title('zero curve (piecewise constant forwards)')
    # plt.grid(True)
    # plt.subplot(1, 2, 2)
    # plt.plot(terms, constant_forwards)
    # plt.xlabel('time')
    # plt.ylabel('f rates')
    # plt.title('instantaneous forward curve (piecewise constant forwards)')
    # plt.grid(True)
    # plt.show()

    zero_rates_df = pd.read_csv('data/zero_rate.csv', index_col=0)/100
    zero_rates = np.array(zero_rates_df.loc['2020-02'])
    terms = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]

    PCF = Raw_interpolation(terms, zero_rates)
    dforwards, zero_curve = PCF.raw_interpolation()

    xx = np.linspace(0, max(terms), 600)
    terms = np.insert(terms, 0, 0)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(xx, zero_curve, 'orange')
    plt.xlabel('time')
    plt.ylabel('zero rates')
    plt.title('zero curve (piecewise constant forwards)')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(terms, dforwards)
    plt.xlabel('time')
    plt.ylabel('f rates')
    plt.title('instantaneous forward curve (piecewise constant forwards)')
    plt.grid(True)
    plt.show()




