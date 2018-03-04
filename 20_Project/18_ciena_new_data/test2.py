import pandas as pd
import numpy as np

filename = '/media/freshield/DATA_U/Ciena_data/new_data/FiberID_6fibers_10Spans/FiberID_6fibers_10Spans_noPCA_10.csv'

reader = pd.read_csv(filename)

print reader.shape
