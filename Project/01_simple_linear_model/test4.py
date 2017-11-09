import pandas as pd
import numpy as np

filename = '/media/freshield/COASAIR1/CIENA/Ciena_data/FiberID_Data.csv'

chunkSize = 10000

reader = pd.read_csv(filename, header=None, iterator=True)

chunk = reader.get_chunk(chunkSize)

loop = True

count = 0

while loop:

    try:
        chunk = reader.get_chunk(chunkSize)

        count += 1
        print count



    except StopIteration:
        print "stop"
        break
