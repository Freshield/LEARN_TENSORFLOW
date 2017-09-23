import pandas as pd

filename = 'data/result_trapha_T_oh.csv'

data = pd.read_csv(filename,header=None)

print data.values[200:210]