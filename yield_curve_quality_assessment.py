# author: Taiga Schwarz
# date modified: 04/27/2022
# Yield Curve Quality Assessment

# functions
def bump(inputs, bp=1):
    """Returns an NxN array of rates, with each row having a rate bump (in bps) applied to the ith input rate, where i
    is equal to the row number (0~N). N is the length of the input rates."""

    import numpy as np

    bump = bp/10000
    N = len(inputs)
    bumped_rates = np.tile(inputs, (N, 1))  # creates an NxN array with each row containing the inputs

    for i in range(N):
        bumped_rates[i,i] = bumped_rates[i,i] + bump

    return bumped_rates

if __name__ == '__main__':
    # imports
    from spline import Spline_fitting
    from piecewise_const_forward import *
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # data
    SOFR_swaps_df = pd.read_csv('data/SOFR_swap.csv').set_index('Date')
    zero_rates_df = pd.read_csv('data/zero_rate.csv', index_col=0)/100

    ############ test localness of the interpolation method ############

    ### piecewise constant forwards
    swaps = np.array(SOFR_swaps_df.loc['2022-02', '1Y':'30Y'])/100
    terms = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])

    # actual zero curve and forward curve
    r_rates1, f_rates1 = piecewise_const_fwd(swaps, terms, 0.5, 0.5)

    # bump the inputs
    bps = 5
    bumped_swaps = bump(swaps, bps)
    r_bumps1 = []
    f_bumps1 = []
    for i in range(len(swaps)):
        r_rates, constant_forwards = piecewise_const_fwd(bumped_swaps[i], terms, 0.5, 0.5)
        r_bumps1.append(r_rates)
        f_bumps1.append(constant_forwards)
    r_bumps = np.array(r_bumps1)
    f_bumps = np.array(f_bumps1)


    # visualize
    tenors = np.arange(terms[0] - 0.5, terms[-1] + 0.5, 0.5)
    for i in range(len(swaps)):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(tenors, r_bumps1[i], 'orange', label='w/ bump')
        plt.plot(tenors, r_rates1, 'blue', label='w/out bump')
        plt.xlabel('tenor')
        plt.ylabel('zero rates')
        plt.title('zero curve (piecewise constant forwards)')
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(terms, f_bumps1[i], 'orange', label='w/ bump')
        plt.plot(terms, f_rates1, 'blue', label='w/out bump')
        plt.xlabel('time')
        plt.ylabel('f rates')
        plt.title('instantaneous forward curve (piecewise constant forwards)')
        plt.grid(True)
        plt.legend()
        plt.suptitle(f'input rate {terms[i]}Y bumped by {bps} bps')
        plt.show()

    ### natural cubic spline
    zero_rates = np.array(zero_rates_df.loc['2022-02'])
    terms = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]

    # actual zero curve and forward curve
    S_actual = Spline_fitting(terms, zero_rates)
    r_rates2, f_rates2 = S_actual.cubic_spline(show_plot=False)

    # bumped rates
    bumped_zero_rates = bump(zero_rates, bps)
    r_bumps2 = []
    f_bumps2 = []
    for i in range(len(zero_rates)):
        S = Spline_fitting(terms, bumped_zero_rates[i])
        zero_rate1, f_rate1 = S.cubic_spline(show_plot=False)
        r_bumps2.append(zero_rate1)
        f_bumps2.append(f_rate1)
    r_bumps2 = np.array(r_bumps2)
    f_bumps2 = np.array(f_bumps2)

    # visualize
    xx = np.linspace(0, max(terms), 600)
    for i in range(len(zero_rates)):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(xx, r_bumps2[i], 'orange', label='w/ bump')
        plt.plot(xx, r_rates2, 'blue', label='w/out bump')
        plt.xlabel('tenor')
        plt.ylabel('zero rates')
        plt.title('zero curve')
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(xx, f_bumps2[i], 'orange', label='w/ bump')
        plt.plot(xx, f_rates2, 'blue', label='w/out bump')
        plt.xlabel('time')
        plt.ylabel('f rates')
        plt.title('instantaneous forward curve')
        plt.grid(True)
        plt.legend()
        plt.suptitle(f'input rate {terms[i]}Y bumped by {bps} bps (natural cubic spline)')
        plt.show()








