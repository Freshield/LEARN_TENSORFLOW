import pandas as pd
import numpy as np

filename = 'result_value.csv'

data = pd.read_csv(filename, header=None).values

print data.shape