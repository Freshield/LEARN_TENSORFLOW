import pandas as pd
import numpy as np

#filename = '/media/freshield/DATA_U/Ciena_data/new_data/FiberID_6fibers_5Spans/FiberID_6fibers_5Spans_noPCA_1.csv'
#filename = '/media/freshield/DATA_U/Ciena_data/new_data/FiberID_6fibers_10Spans/FiberID_6fibers_10Spans_noPCA_1.csv'
filename = '/media/freshield/DATA_U/Ciena_data/new_data/FiberID_6fibers_20Spans/FiberID_6fibers_20Spans_noPCA_1.csv'

chunkSize = 10
savename = '20spans_sample.csv'

reader = pd.read_csv(filename, iterator=True)

chunk = reader.get_chunk(chunkSize)

np.savetxt(savename, chunk, delimiter=",")

print chunk.shape
