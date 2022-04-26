#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 12:51:29 2022

@author: Caleb
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
from scipy.optimize import minimize

import os
os.chdir('/Users/Caleb/Documents/Classes/Boston/Spring 2022/728/FinalProject')

from piecewise_const_forward import *


# What curves look like naturally
swap_df = pd.read_csv('SOFR_swap.csv')
sofr_swaps = np.array(swap_df.iloc[0, :])[3:]  # starting at the 1 year swap

# sofr_swaps[0] = 2.4217
# sofr_swaps[1] = 2.3912


terms = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])
zero_rates, constant_forwards = piecewise_const_fwd(sofr_swaps, terms, 0.5, 0.5)
tenors = np.arange(terms[0] - 0.5, terms[-1] + 0.5, 0.5)

plt.plot(terms, constant_forwards, label='forward curve')
plt.plot(tenors, zero_rates, label='zero curve')
plt.xlabel('time (years)')
plt.title('Yield Curves by Raw Interpolation Method (as of 2022-03)')
plt.legend()
plt.show()


# asdf = piecewise_const_fwd(bull_steepener(4, 0.01, 5, sofr_swaps), terms, 0.5, 0.5)[0]


# bumps up the ith term rates
def bump(i, diff, curve):
    asdf = curve.copy()
    asdf[i] = asdf[i] +  diff
    return asdf




# Bull steepener. For bear steepener set negative slope
def triangle(i, fixed_height, apex, curve):
    # slope is in terms of basis points
            
    
    shift = np.zeros(15)
    if i == 0:
        shift[0] = apex
        shift[1] = fixed_height
    elif i == len(curve) - 1:
        shift[len(curve) - 2] = fixed_height
        shift[len(curve) - 1] = apex
        
    else:
        shift[i - 1] = fixed_height
        shift[i] = apex
        shift[i + 1] = fixed_height
        
    return curve + shift



triangle(11, 0.1, 0.2, sofr_swaps )
triangle(10, 0.1, 0.2, sofr_swaps )
triangle(9, 0.1, 0.2, sofr_swaps )
triangle(8, 0.1, 0.2, sofr_swaps )
triangle(7, 0.1, 0.2, sofr_swaps )


# Gets a vector of prices of each of the bootstrapping instruments (the sofr_swaps rates)
def instrument_prices(terms, rates):
    prices = np.zeros(len(terms))
    for i in range(len(terms)):
        prices[i] = np.exp(-terms[i] * rates[i]/100)
    return prices



# See the effects of bumps
bp = 0.01
for i in range(0, 15):
    sofr_swaps = np.array(swap_df.iloc[0, :])[3:]
    sofr_swaps = bump(i, bp, sofr_swaps)
    zero_rates, constant_forwards = piecewise_const_fwd(sofr_swaps, terms, 0.5, 0.5)
    plt.plot(terms, constant_forwards, label='forward curve')
    plt.plot(tenors, zero_rates, label='zero curve')
    plt.xlabel('time (years)')
    plt.title('Effect of Bumps on Yield Curves by Raw Interpolation Method (as of 2022-03)')
    plt.legend()
    plt.show()
    
    
# see the effects of triangle
bp = 0.01
for i in range(0, 15):
    sofr_swaps = np.array(swap_df.iloc[0, :])[3:]
    sofr_swaps = triangle(i, 0.01, 0.05, sofr_swaps )
    zero_rates, constant_forwards = piecewise_const_fwd(sofr_swaps, terms, 0.5, 0.5)
    plt.plot(tenors, zero_rates, label='zero curve')
    plt.xlabel('time (years)')
    plt.title('Effect of Steepeners on Yield Curves by Raw Interpolation Method (as of 2022-03)')
    plt.legend()
    plt.show()
    
# see the effects of pure bumps
bp = 0.1
for i in range(0, 15):
    print(i)
    sofr_swaps = np.repeat(1, 15)
    sofr_swaps = bump(i, bp, sofr_swaps)
    plt.plot(terms, sofr_swaps, label='curve')
    plt.xlabel('time (years)')
    plt.title('Visualization of Effect of Steepener')
    plt.legend()
    plt.show()
    
    
# see the effects of pure triangle
bp = 0.1
for i in range(0, 15):
    print(i)
    sofr_swaps = np.repeat(1, 15)
    sofr_swaps = triangle(i, 0.1, 0.2, sofr_swaps)
    plt.plot(terms, sofr_swaps, label='curve')
    plt.xlabel('time (years)')
    plt.title('Visualization of Effect of Steepener')
    plt.legend()
    plt.show()





'''
STEEPENER
'''


# Creating a steepener by shifting sofr swap rates
bear_steepener = sofr_swaps.copy()
for i in range(len(terms) - 2):
    bear_steepener[i + 2] += 0.05 * i

# getting shifted zero rates and constant forwards with steepened sofr rates
zero_rates, constant_forwards = piecewise_const_fwd(bear_steepener, terms, 0.5, 0.5)


# Visualizing the difference between steepened and non steepened sofr swaps
plt.plot(terms, bear_steepener, label='bear steepener')
plt.plot(terms, sofr_swaps, label='sofr_swaps')
plt.xlabel('time (years)')
plt.title('bear steepener of sofr swap rates (as of 2022-03)')
plt.legend()
plt.show()


# Plotting steepened constant and zero rates. Compare these with before steepening (from up there)
plt.plot(terms, constant_forwards, label='forwards curve')
plt.plot(tenors, zero_rates, label='zero curve')
plt.xlabel('time (years)')
plt.title('zero and forwards after steepening (as of 2022-03)')
plt.legend()
plt.show()
    

    
    
# hedging a steepened portfolio. Plot the portfolio changes we need to make like in hagan and west?

# Hedging part PQ = deltaV
# P is prices matrix where Pij is the change in price of the jth bootstrapping instrument under the ith curve
# Q is the solution matrix, the adjustments to our portfoliio
# delta V is the change in our risk instrument



# Creates a matrix of the zero rate instrument prices, where rows are all the zero rate prices for each shift, and 
# columns represent indices or maturities of the zero rate instruments 
zero_rates, constant_forwards = piecewise_const_fwd(sofr_swaps, terms, 0.5, 0.5)
steepened_zero_rates, steepened_constant_forwards = piecewise_const_fwd(bear_steepener, terms, 0.5, 0.5)


# getting the difference in portfolio value before and after steepening, given an equally weighted portfolio
# on the zero coupon bonds


# Initially equally weighted portfolio
init_weights = np.repeat(1/15, 15)
init_port = instrument_prices(np.arange(1, 16), zero_rates[terms * 2 - 1] )
steepened_port = instrument_prices(np.arange(1, 16), steepened_zero_rates[terms * 2 - 1] )
deltaV = np.dot(init_weights, steepened_port) - np.dot(init_weights, init_port)


# Create a matrix of the sofr rate instrument prices, where rows are the changes in prices given new rates, and 
# columns represent indices or maturities of the sofr rate instruments
price_diff = steepened_port - init_port


def pvoptim(weights):
    diff = np.dot(init_weights, init_port) - np.dot(weights, steepened_port)
    return abs(diff)


# Performing present value matching, and plotting
new_weights = minimize(pvoptim, np.repeat(1/15, 15),  method='Nelder-Mead', tol=1e-20).x
plt.plot(terms, init_weights, label='before steepener')
plt.plot(terms, new_weights, label='after steepener')
plt.xlabel('bond maturity (years)')
plt.title('Portfolio weights after steepening (as of 2022-03)')
plt.legend()
plt.show()
    

# Performing duration matching, using key rate duration
original_weights = init_port/sum(init_port)
init_duration = sum(terms * original_weights)
steepened_weights = steepened_port/sum(steepened_port)
steepened_duration = sum(terms * steepened_weights)


# Optimizes by matching duration and present value at the same time
def duroptim(weights):
    diff1 = np.dot(init_weights, init_port) - np.dot(weights, steepened_port)
    diff2 = init_duration - sum(terms * weights)
    return abs(diff1 + diff2)


hedge_weights = minimize(duroptim, np.repeat(1/15, 15),  method='Nelder-Mead', tol=1e-20).x
plt.plot(terms, init_weights, label='before steepener')
plt.plot(terms, hedge_weights, label='after steepener')
plt.xlabel('bond maturity (years)')
plt.title('Portfolio weights after steepening (as of 2022-03)')
plt.legend()
plt.show()

    
# This hedge does not look too reasonable likely because of the spike in the beginning
# PQ = deltaV
P = price_mat
Pinv = scipy.linalg.inv(P)
Q = np.dot(Pinv,  deltaV)
plt.plot(Q)

# bumping by 0.4 creates sick plot here of prices against maturities for zero rate instruments
plt.plot(np.arange(1, 61), port_mat.T)



