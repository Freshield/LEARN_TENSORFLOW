import pandas as pd
import numpy as np

filename = '/media/freshield/New_2T_Data/Ciena_new_data/20spans/norm/Raw_data_0_train.csv'

reader = pd.read_csv(filename, header=None, chunksize=100)

count = 0

for chunk in reader:
    # print '    line %d begin convert' % count
    if count % 50 == 0:
        print '%s done %d lines convert' % (filename.split('/')[-1], count)

    data = chunk.values
    for lineNum in range(data.shape[0]):
        line = data[lineNum]
        print line.shape
        break

    break