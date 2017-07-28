import pandas as pd
import numpy as np

data = pd.read_csv('data/train.csv', header=None)

print data.values.shape