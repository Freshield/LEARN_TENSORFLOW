import pandas as pd
import numpy as np

filename = '/media/freshield/New_2T_Data/Ciena_new_data/20spans/norm/Raw_data_999_test.csv'

data = pd.read_csv(filename, header=None)

print data.shape