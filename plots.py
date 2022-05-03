# author: Taiga Schwarz
# date modified: 05/02/2022
# useful plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### plot 1 -- 03/2022 zero rates that were bootstrapped from U.S. Treasury securities (source: Bloomberg)
# data
zero_rates_df = pd.read_csv('data/zero_rate.csv', index_col=0)/100
dates = zero_rates_df.columns
zero_rates = np.array(zero_rates_df.loc['2022-03'])
terms = np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30])

# plot
plt.scatter(terms, zero_rates, marker='o')
plt.xlabel("Maturity")
plt.ylabel("Zero Rate")
plt.xticks(terms, dates)
plt.title("Zero Rates Bootstrapped from U.S. Treasury securities (03/2022)")
plt.show()

print(zero_rates_df.iloc[0,:])


