# authors: Taiga Schwarz and Caleb Qi
# date modified: 04/29/2022

###### bump hedging -- hedge against small changes in the input rates #######

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# data
zero_rates_df = pd.read_csv('data/zero_rate.csv', index_col=0)/100
zero_rates = np.array(zero_rates_df.loc['2022-03'])
terms = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]

### raw interpolation method

# actual zero curve and forward curve
PCF = Raw_interpolation(terms, zero_rates)
f_curve1, zero_curve1 = PCF.raw_interpolation()

# bump the inputs
bps = 1
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

