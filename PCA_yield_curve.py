# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 00:13:15 2022

@author: Dieynaba Awa Ndiaye
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns



#import swaps
swap_df = pd.read_csv('SOFR_swap.csv')
swap_df.head()


#plot swap rate
swap_df2 = swap_df.copy()
swap_df.plot(figsize=(10,5))
plt.ylabel("Rate")
plt.legend(bbox_to_anchor=(1.01, 0.9), loc=2)
plt.show()


#plot heatmap
sns.heatmap(swap_df.corr())
plt.show()


#PCA covariance method
def PCA(df, num_reconstruct):
    
    df -= df.mean(axis=0)
    df = df.iloc[:, 1:]
    R = np.cov(df, rowvar=False)
    eigenvals, eigenvecs = sp.linalg.eigh(R)
    eigenvecs = eigenvecs[:, np.argsort(eigenvals)[::-1]]
    eigenvals = eigenvals[np.argsort(eigenvals)[::-1]]
    eigenvecs = eigenvecs[:, :num_reconstruct]

    return np.dot(eigenvecs.T, df.T).T, eigenvals, eigenvecs

scores, evals, evecs = PCA(swap_df, 7)


#plot PC(1), PC(2), PC(3)
#PC(1) Parallel shifts in yield curve (shifts across the entire yield curve)
#PC(2) Changes in short/long rates (i.e. steepening/flattening of the curve)
#PC(3) Changes in curvature of the model (twists)
evecs = pd.DataFrame(evecs)
plt.plot(evecs.iloc[:, 0:3])
plt.show()









