import pandas as pd
import numpy as np

filename = '/media/freshield/DATA_U/Ciena_data/new_data/FiberID_6fibers_20Spans/FiberID_6fibers_20Spans_noPCA_10.csv'

reader = pd.read_csv(filename, header=None, iterator=True, dtype=np.float32)

loop = True

count = 0

while loop:
    try:
        chunk = reader.get_chunk(1000)

        print count

        print chunk.shape

        # i += chunk.shape[0]
        count += 1



    except StopIteration:
        print "stop"
        print count
        break