import pandas as pd

filename = '10spans_sample.csv'

filename1 = '/media/freshield/CORSAIR/CIENA/Ciena_data/ciena1000.csv'

data = pd.read_csv(filename)

print data.shape

print data.values[:,12001]

data = pd.read_csv(filename1)

print data.shape

print data.values[0:10,6200:6211]