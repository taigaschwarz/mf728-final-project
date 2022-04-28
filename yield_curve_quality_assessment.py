# author: Taiga Schwarz
# date modified: 04/27/2022
# Yield Curve Quality Assessment

# functions
def bumps(inputs, bp=1):
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
    import matplotlib.pyplot as plt
    from monotone_convex import Monotone_convex
    import numpy as np
    import pandas as pd
    from piecewise_const_forward import *
    from spline import Spline_fitting

    # data
    SOFR_swaps_df = pd.read_csv('data/SOFR_swap.csv').set_index('Date')
    zero_rates_df = pd.read_csv('data/zero_rate.csv', index_col=0)/100

    ############ test localness of the interpolation method ############

    zero_rates = np.array(zero_rates_df.loc['2022-03'])
    terms = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]

    ### piecewise constant forwards
    # swaps = np.array(SOFR_swaps_df.loc['2022-02', '1Y':'30Y'])/100
    # terms = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])

    # actual zero curve and forward curve
    PCF = Raw_interpolation(terms, zero_rates)
    f_rates1, r_rates1 = PCF.raw_interpolation()
    # r_rates1, f_rates1 = piecewise_const_fwd(swaps, terms, 0.5, 0.5)

    # bump the inputs
    bps = 5
    bumped_zero_rates = bumps(zero_rates, bps)
    r_bumps1 = []
    f_bumps1 = []
    for i in range(len(zero_rates)):
        PCF = Raw_interpolation(terms, bumped_zero_rates[i])
        constant_forwards, r_rates = PCF.raw_interpolation()
        r_bumps1.append(r_rates)
        f_bumps1.append(constant_forwards)
    r_bumps1 = np.array(r_bumps1)
    f_bumps1 = np.array(f_bumps1)

    # visualize
    xx = np.linspace(0, max(terms), 600)
    tenors = np.insert(terms, 0, 0)
    for i in range(1,len(zero_rates)):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(xx, r_bumps1[i], 'orange', label='w/ bump')
        plt.plot(xx, r_rates1, 'blue', label='w/out bump')
        plt.xlabel('time')
        plt.ylabel('zero rates')
        plt.title('zero curve (piecewise constant forwards)')
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(tenors, f_bumps1[i], 'orange', label='w/ bump')
        plt.plot(tenors, f_rates1, 'blue', label='w/out bump')
        plt.xlabel('time')
        plt.ylabel('f rates')
        plt.title('instantaneous forward curve (piecewise constant forwards)')
        plt.grid(True)
        plt.legend()
        plt.suptitle(f'input rate {terms[i]}Y bumped by {bps} bps')
        plt.show()

    ### natural cubic spline

    # actual zero curve and forward curve
    S_actual = Spline_fitting(terms, zero_rates)
    r_rates2, f_rates2 = S_actual.cubic_spline(show_plot=False)

    # bumped rates
    bumped_zero_rates = bumps(zero_rates, bps)
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

    ### cubic B-spline

    # actual zero curve and forward curve
    S_actual = Spline_fitting(terms, zero_rates)
    r_rates3, f_rates3 = S_actual.Bspline(show_plot=False)

    # bumped rates
    r_bumps3 = []
    f_bumps3 = []
    for i in range(len(zero_rates)):
        S = Spline_fitting(terms, bumped_zero_rates[i])
        zero_rate3, f_rate3 = S.Bspline(show_plot=False)
        r_bumps3.append(zero_rate3)
        f_bumps3.append(f_rate3)
    r_bumps3 = np.array(r_bumps3)
    f_bumps3 = np.array(f_bumps3)

    # visualize
    xx = np.linspace(0, max(terms), 600)
    for i in range(len(zero_rates)):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(xx, r_bumps3[i], 'orange', label='w/ bump')
        plt.plot(xx, r_rates3, 'blue', label='w/out bump')
        plt.xlabel('tenor')
        plt.ylabel('zero rates')
        plt.title('zero curve')
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(xx, f_bumps3[i], 'orange', label='w/ bump')
        plt.plot(xx, f_rates3, 'blue', label='w/out bump')
        plt.xlabel('time')
        plt.ylabel('f rates')
        plt.title('instantaneous forward curve')
        plt.grid(True)
        plt.legend()
        plt.suptitle(f'input rate {terms[i]}Y bumped by {bps} bps (cubic B-spline)')
        plt.show()

    ### monotone convex method (Hagan-West)

    # actual curves
    M_actual = Monotone_convex(terms, zero_rates)
    r_rates4, f_rates4 = M_actual.fitting()

    # bumped rates
    r_bumps4 = []
    f_bumps4 = []
    for i in range(len(zero_rates)):
        M = Monotone_convex(terms, bumped_zero_rates[i])
        zero_rate4, f_rate4 = M.fitting()
        r_bumps4.append(zero_rate4)
        f_bumps4.append(f_rate4)
    r_bumps4 = np.array(r_bumps4)
    f_bumps4 = np.array(f_bumps4)

    # visualize
    xx = np.linspace(0, max(terms), 600)
    for i in range(len(zero_rates)):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(xx, r_bumps4[i], 'orange', label='w/ bump')
        plt.plot(xx, r_rates4, 'blue', label='w/out bump')
        plt.xlabel('tenor')
        plt.ylabel('zero rates')
        plt.title('zero curve')
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(xx, f_bumps4[i], 'orange', label='w/ bump')
        plt.plot(xx, f_rates4, 'blue', label='w/out bump')
        plt.xlabel('time')
        plt.ylabel('f rates')
        plt.title('instantaneous forward curve')
        plt.grid(True)
        plt.legend()
        plt.suptitle(f'input rate {terms[i]}Y bumped by {bps} bps (monotone convex)')
        plt.show()

    # ############ test stability of the method over time ############
    #
    # ### piecewise constant forwards
    # swaps_ts = np.array(SOFR_swaps_df.loc['2022-03':'2021-10', '1Y':'30Y']) / 100
    # dates = SOFR_swaps_df.loc['2022-03':'2021-10', '1Y':'30Y'].index
    # terms = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])
    # zero_rates1_stab = []
    # f_rates1_stab = []
    # for i in range(len(swaps_ts)):
    #     r, f = piecewise_const_fwd(swaps_ts[i], terms, 0.5, 0.5)
    #     zero_rates1_stab.append(r)
    #     f_rates1_stab.append(f)
    #
    # # visualize
    # tenors = np.arange(terms[0] - 0.5, terms[-1] + 0.5, 0.5)
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # for i in range(len(swaps_ts)):
    #     plt.plot(tenors, zero_rates1_stab[i], label=f'{dates[i]}')
    # plt.xlabel('tenor')
    # plt.ylabel('zero rates')
    # plt.title('zero curve')
    # plt.grid(True)
    # plt.ylim((0, 0.03))
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # for i in range(len(swaps_ts)):
    #     plt.plot(terms, f_rates1_stab[i], label=f'{dates[i]}')
    # plt.xlabel('time')
    # plt.ylabel('f rates')
    # plt.title('instantaneous forward curve')
    # plt.grid(True)
    # plt.ylim((0, 0.04))
    # plt.legend()
    # plt.suptitle('Change in yield curve over 03/2022 ~ 11/2021 (piecewise constant forwards)')
    # plt.show()
    #
    # ### natural cubic spline
    # zero_rates_ts = np.array(zero_rates_df.loc['2022-03':'2021-10'])
    # terms = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    # zero_rates2_stab = []
    # f_rates2_stab = []
    # for i in range(len(zero_rates_ts)):
    #     S = Spline_fitting(terms, zero_rates_ts[i])
    #     r, f = S.cubic_spline(show_plot=False)
    #     zero_rates2_stab.append(r)
    #     f_rates2_stab.append(f)
    #
    # # visualize
    # xx = np.linspace(0, max(terms), 600)
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # for i in range(len(zero_rates_ts)):
    #     plt.plot(xx, zero_rates2_stab[i], label=f'{dates[i]}')
    # plt.xlabel('tenor')
    # plt.ylabel('zero rates')
    # plt.title('zero curve')
    # plt.grid(True)
    # plt.ylim((0, 0.03))
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # for i in range(len(zero_rates_ts)):
    #     plt.plot(xx, f_rates2_stab[i], label=f'{dates[i]}')
    # plt.xlabel('time')
    # plt.ylabel('f rates')
    # plt.title('instantaneous forward curve')
    # plt.grid(True)
    # plt.ylim((0, 0.04))
    # plt.legend()
    # plt.suptitle('Change in yield curve over 03/2022 ~ 11/2021 (natural cubic spline)')
    # plt.show()
    #
    # ### cubic B-spline
    # zero_rates3_stab = []
    # f_rates3_stab = []
    # for i in range(len(zero_rates_ts)):
    #     S = Spline_fitting(terms, zero_rates_ts[i])
    #     r, f = S.Bspline(show_plot=False)
    #     zero_rates3_stab.append(r)
    #     f_rates3_stab.append(f)
    #
    # # visualize
    # xx = np.linspace(0, max(terms), 600)
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # for i in range(len(zero_rates_ts)):
    #     plt.plot(xx, zero_rates3_stab[i], label=f'{dates[i]}')
    # plt.xlabel('tenor')
    # plt.ylabel('zero rates')
    # plt.title('zero curve')
    # plt.grid(True)
    # plt.ylim((0, 0.03))
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # for i in range(len(zero_rates_ts)):
    #     plt.plot(xx, f_rates3_stab[i], label=f'{dates[i]}')
    # plt.xlabel('time')
    # plt.ylabel('f rates')
    # plt.title('instantaneous forward curve')
    # plt.grid(True)
    # plt.ylim((0, 0.04))
    # plt.legend()
    # plt.suptitle('Change in yield curve over 03/2022 ~ 11/2021 (cubic B-spline)')
    # plt.show()
    #
    #
    #
