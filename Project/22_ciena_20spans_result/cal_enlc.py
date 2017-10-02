import pandas as pd
import numpy as np

filename = 'result_probability.csv'
label = 'result_label.csv'

data = pd.read_csv(filename, header=None).values
label_v = pd.read_csv(label, header=None).values

ENLC_array = np.array([34.515,23.92,21.591,25.829,28.012,29.765]).reshape((6,1))

result = np.matmul(data,ENLC_array)
result_label = np.matmul(label_v, ENLC_array)

#np.savetxt('enlc_value.csv', result, delimiter=',')
#np.savetxt('enlc_label.csv', result_label, delimiter=',')
diff = result - result_label
np.savetxt('enlc_diff.csv', diff, delimiter=',')