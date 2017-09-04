import pandas as pd
import numpy as np

filename = '/media/freshield/New_2T_Data/Ciena_new_data/20spans/norm/Raw_data_0_test.csv'

reader = pd.read_csv(filename, header=None, chunksize=400)

count = 0

for chunk in reader:

    data = chunk.values
    print data.shape
    for lineNum in range(data.shape[0]):
        line = data[lineNum]
