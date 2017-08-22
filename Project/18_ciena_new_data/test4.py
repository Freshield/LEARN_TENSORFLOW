import pandas as pd
import numpy as np

"""
(-0.0095194001, -0.034023002, 4036.5, 31.327999, -1.1167001)
(0.010286, 0.0053022001, 9904.0996, 127.93, 5.3010998)
"""

filename = '/media/freshield/LINUX/Ciena/norm/Raw_data_0_train.csv'

data = pd.read_csv(filename,header=None)

print np.mean(data.values[:,12001:12011])