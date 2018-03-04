import pandas as pd
import numpy as np

filename1 = 'result_value.csv'
filename2 = 'data/clean_raw_test_35.csv'

data1 = pd.read_csv(filename1).values
data2 = pd.read_csv(filename2).values

right_num = 0.0

for i in range(data1.shape[0]):
    if data1[i,-1] == 1:
        print data1[i]
        if data1[i,0] == 1:
            right_num += 1

print right_num / data1.shape[0]