# authors: Taiga Schwarz and Caleb Qi
# date modified: 04/29/2022

###### bump hedging -- hedge against small changes in the input rates #######

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import pandas as pd
from monotone_convex import Monotone_convex
from piecewise_const_forward import Raw_interpolation
from spline import Spline_fitting


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

def ZCB_price(r, t, F):
    """Computes the price of a zero-coupon bond at rate r, maturity t, and face F."""

    import numpy as np

    price = F * np.exp(-r*t)

    return price

# data
zero_rates_df = pd.read_csv('data/zero_rate.csv', index_col=0)/100
dates = zero_rates_df.columns
zero_rates = np.array(zero_rates_df.loc['2018-03'])
terms = np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30])

### global variables
t = 4.5  # the tenor of the swap we are hedging
F = 1  # notional

### bump the inputs
bps = 1
bumped_zero_rates = bumps(zero_rates, bps)

### original prices of the input ZCB's
og_prices = ZCB_price(zero_rates, terms, F)

### construct matrix P, where P_ij is the change in price of the jth bootstrapping instrument of the ith curve
P = np.zeros((len(zero_rates), len(zero_rates)))
for i in range(len(zero_rates)):
    new_prices = ZCB_price(bumped_zero_rates[i], terms, F)
    P[i] = new_prices - og_prices


### raw interpolation method

# actual zero curve and forward curve
PCF = Raw_interpolation(terms, zero_rates)
f_curve1, zero_curve1 = PCF.raw_interpolation()
swap_price1 = PCF.raw_swap_price(t) * F  # F is the notional

# curves constructed from the bumped inputs
r_bumps1 = []
f_bumps1 = []
swap_prices1 = []
for i in range(len(zero_rates)):
    PCF = Raw_interpolation(terms, bumped_zero_rates[i])
    constant_forwards, r_rates = PCF.raw_interpolation()
    swap_price = PCF.raw_swap_price(t) * F
    r_bumps1.append(r_rates)
    f_bumps1.append(constant_forwards)
    swap_prices1.append(swap_price)
r_bumps1 = np.array(r_bumps1)
f_bumps1 = np.array(f_bumps1)
swap_prices1 = np.array(swap_prices1)

# difference between the new and old price -- vector dV
swap_price_og = np.repeat(swap_price1, len(zero_rates))
dV1 = swap_prices1 - swap_price_og
print("dV1: ", dV1)

# solve PQ = dV for the hedge vector Q
Q1 = inv(P) @ dV1
print('Q1: ', Q1)

# plot
asset_num = np.arange(1, len(zero_rates) + 1)
plt.bar(asset_num, Q1.flatten())
plt.xlabel("Maturity")
plt.ylabel("Notional Amount")
plt.xticks(asset_num, dates)
plt.title(f"Hedging Portfolio for a {t}Y Swap (Raw Interpolation)")
plt.show()


### natural cubic spline interpolation method

# actual zero curve and forward curve
S = Spline_fitting(terms, zero_rates)
zero_curve2, f_curve2 = S.cubic_spline()
swap_price2 = S.cubic_spline_swap_price(t) * F  # F is the notional

# curves constructed from the bumped inputs
r_bumps2 = []
f_bumps2 = []
swap_prices2 = []
for i in range(len(zero_rates)):
    S = Spline_fitting(terms, bumped_zero_rates[i])
    r_rates, f_rates = S.cubic_spline()
    swap_price = S.cubic_spline_swap_price(t) * F
    r_bumps2.append(r_rates)
    f_bumps2.append(f_rates)
    swap_prices2.append(swap_price)
r_bumps2 = np.array(r_bumps2)
f_bumps2 = np.array(f_bumps2)
swap_prices2 = np.array(swap_prices2)

# difference between the new and old price -- vector dV
swap_price_og = np.repeat(swap_price2, len(zero_rates))
dV2 = swap_prices2 - swap_price_og

# solve PQ = dV for the hedge vector Q
Q2 = inv(P) @ dV2

# plot
asset_num = np.arange(1, len(zero_rates) + 1)
plt.bar(asset_num, Q2.flatten())
plt.xlabel("Maturity")
plt.ylabel("Notional Amount")
plt.xticks(asset_num, dates)
plt.title(f"Hedging Portfolio for a {t}Y Swap (Natural Cubic Spline)")
plt.show()


### cubic B-spline interpolation method

# actual zero curve and forward curve
zero_curve3, f_curve3 = S.Bspline()
swap_price3 = S.Bspline_swap_price(t) * F  # F is the notional

# curves constructed from the bumped inputs
r_bumps3 = []
f_bumps3 = []
swap_prices3 = []
for i in range(len(zero_rates)):
    S = Spline_fitting(terms, bumped_zero_rates[i])
    r_rates, f_rates = S.Bspline()
    swap_price = S.Bspline_swap_price(t) * F
    r_bumps3.append(r_rates)
    f_bumps3.append(f_rates)
    swap_prices3.append(swap_price)
r_bumps3 = np.array(r_bumps3)
f_bumps3 = np.array(f_bumps3)
swap_prices3 = np.array(swap_prices3)

# difference between the new and old price -- vector dV
swap_price_og = np.repeat(swap_price3, len(zero_rates))
dV3 = swap_prices3 - swap_price_og

# solve PQ = dV for the hedge vector Q
Q3 = inv(P) @ dV3

# plot
asset_num = np.arange(1, len(zero_rates) + 1)
plt.bar(asset_num, Q3.flatten())
plt.xlabel("Maturity")
plt.ylabel("Notional Amount")
plt.xticks(asset_num, dates)
plt.title(f"Hedging Portfolio for a {t}Y Swap (Cubic B-Spline)")
plt.show()

# ### monotone convex interpolation method
#
# # actual zero curve and forward curve
# M = Monotone_convex(terms, zero_rates)
# zero_curve4, f_curve4 = M.fitting()
# swap_price4 = M.mc_swap_price(t) * F  # F is the notional
#
# # curves constructed from the bumped inputs
# r_bumps4 = []
# f_bumps4 = []
# swap_prices4 = []
# for i in range(len(zero_rates)):
#     M = Monotone_convex(terms, bumped_zero_rates[i])
#     r_rates, f_rates = M.fitting()
#     swap_price = M.mc_swap_price(t) * F
#     r_bumps4.append(r_rates)
#     f_bumps4.append(f_rates)
#     swap_prices4.append(swap_price)
# r_bumps4 = np.array(r_bumps4)
# f_bumps4 = np.array(f_bumps4)
# swap_prices4 = np.array(swap_prices4)
#
# # difference between the new and old price -- vector dV
# swap_price_og = np.repeat(swap_price4, len(zero_rates))
# dV4 = swap_prices4 - swap_price_og
#
# # solve PQ = dV for the hedge vector Q
# Q4 = inv(P) @ dV4
#
# # plot
# asset_num = np.arange(1, len(zero_rates) + 1)
# plt.bar(asset_num, Q4.flatten())
# plt.xlabel("Maturity")
# plt.ylabel("Notional Amount")
# plt.xticks(asset_num, dates)
# plt.title(f"Hedging Portfolio for a {t}Y Swap (Monotone Convex)")
# plt.show()




