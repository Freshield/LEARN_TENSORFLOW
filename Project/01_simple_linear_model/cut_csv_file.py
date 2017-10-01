import pandas as pd

filename = '/home/freshield/ciena_test/FiberID_Data.csv'
target_file_name = 'ciena_test.csv'
number = 10000

dataset = pd.read_csv(filename, header=None)
data = dataset.head(number)

data.to_csv(target_file_name, header=None, index=None)

test = pd.read_csv(target_file_name, header=None)

print test.shape