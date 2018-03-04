import numpy as np
import pandas as pd

file_name = 'data_sample.csv'

file = pd.read_csv(file_name, header=None)

print file.values.shape
