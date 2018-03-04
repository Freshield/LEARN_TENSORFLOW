import pandas as pd
import numpy as np

filename = '20spans_sample.csv'

data = pd.read_csv(filename, header=None).values

print data[1,24041:24061]
print data[1,24061:]
