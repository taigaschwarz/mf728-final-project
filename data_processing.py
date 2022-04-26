"""
Program:
Author: cai
Date: 2022-04-17
"""
import os
import math
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 3000)


# data processing
def get_file_name(base):
    def find_AllFile(base):
        for root, ds, fs in os.walk(base):
            for f in fs:
                fullname = os.path.join(root, f)
                yield fullname
    file_names = []
    for i in find_AllFile(base):
        file_names.append(i)
    for i in file_names:
        if '.DS_Store' in i:
            file_names.remove(i)
    return file_names


def data_processing(file_names):
    data = pd.DataFrame()
    for i in file_names:
        temp = pd.read_excel(i, index_col=0)
        temp.columns = [i[-8:-5]]
        data = pd.concat([data, temp], axis=1)
    data.index = pd.to_datetime(data.index)
    data.index = data.index.strftime('%Y-%m')
    data.columns = ['15Y','8Y','10Y','30Y','9Y','5Y','7Y','3Y','1Y','6Y','4Y','20Y','2Y','6M']
    data = data[['6M','1Y','2Y','3Y','4Y','5Y','6Y','7Y','8Y','9Y','10Y','15Y','20Y','30Y']]
    return data


path = get_file_name('/Users/cai/python_program/MF728/data/zero_rate')
data = data_processing(path)
print(data)
data.to_csv("zero_rate.csv")


# B-splines

# class B_splines:
#
#     def __init__(self, ):


# if __name__ == "__main__":
#
#     data = pd.read_csv('/Users/cai/python_program/MF728/data/SOFR_swap.csv', index_col=0)
#     tenor = [0.25,0.5,1,2,3,4,5,6,7,8,9,10,12,15,20,25,30]
#     # print(data)
#     temp_data = np.array(data.iloc[:1]).flatten()
#     print(temp_data)
#     # yy = cubic_splines(tenor, temp_data)
#     CS = CubicSplines(tenor, temp_data)
#     print(CS.get_delta())







