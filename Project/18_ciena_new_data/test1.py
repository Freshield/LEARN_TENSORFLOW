import pandas as pd
import numpy as np

filename = '/media/freshield/DATA_U/Ciena_data/new_data/FiberID_6fibers_10Spans/FiberID_6fibers_10Spans_noPCA_10.csv'

reader = pd.read_csv(filename, header=None, iterator=True, dtype=np.float32)

chunkSize = 1000

loop = True

count = 0

while loop:
    try:

        chunk = reader.get_chunk(chunkSize)
        print count
        print chunk.shape

        # i += chunk.shape[0]
        count += 1


    except StopIteration:
        print "stop"
        break

print count