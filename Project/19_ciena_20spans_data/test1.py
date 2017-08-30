import pandas as pd
import numpy as np

filename = '/media/freshield/BACKUP/FiberID_6fibers_20Spans_noPCA_1.csv'

reader = pd.read_csv(filename, header=None, iterator=True, dtype=np.float32)

chunk = reader.get_chunk(10)

print chunk.shape

np.savetxt('sample/sample_set.csv', chunk, delimiter=",")

print 'done save sample'