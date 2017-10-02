import pandas as pd
import numpy as np

filename = 'label_F.csv'

data = pd.read_csv(filename, header=None, dtype=np.int32).values

print data