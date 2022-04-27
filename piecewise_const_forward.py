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
    intervals = np.diff(np.insert(terms, 0, 0))  # time intervals between input swap term maturity dates

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
    tenors = np.arange(terms[0] - 0.5, terms[-1] + 0.5, 0.5)
    zero_rates = df_to_zero(discount_factors, tenors)

    return zero_rates, constant_forwards


if __name__=='__main__':

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    swap_df = pd.read_csv('data/SOFR_swap.csv')
    sofr_swaps = np.array(swap_df.iloc[1, :])[3:]/100  # starting at the 1 year swap
    terms = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])
    # sofr_swaps = np.array([2.8438, 3.060, 3.126, 3.144, 3.150, 3.169, 3.210, 3.237])/100
    # terms = np.array([1, 2, 3, 4, 5, 7, 10, 30])

    zero_rates, constant_forwards = piecewise_const_fwd(sofr_swaps, terms, 0.5, 0.5)

    tenors = np.arange(terms[0] - 0.5, terms[-1] + 0.5, 0.5)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(tenors, zero_rates, 'orange')
    plt.xlabel('tenor')
    plt.ylabel('zero rates')
    plt.title('zero curve (piecewise constant forwards)')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(terms, constant_forwards)
    plt.xlabel('time')
    plt.ylabel('f rates')
    plt.title('instantaneous forward curve (piecewise constant forwards)')
    plt.grid(True)
    plt.show()