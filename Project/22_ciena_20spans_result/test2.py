import pandas as pd
import numpy as np

filename1 = 'result_label.csv'
filename2 = 'result_value.csv'

data1 = pd.read_csv(filename1,header=None)
data2 = pd.read_csv(filename2,header=None)

print data1.values.shape
print data2.values.shape