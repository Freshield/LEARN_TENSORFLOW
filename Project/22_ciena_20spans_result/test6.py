import pandas as pd
import numpy as np

label_name = 'enlc_label.csv'
value_name = 'enlc_value.csv'
diff_name = 'enlc_diff.csv'

label_v = pd.read_csv(label_name,header=None).values
value_v = pd.read_csv(value_name,header=None).values
diff_v = pd.read_csv(diff_name,header=None).values

length = value_v.shape[0]

label_v = np.reshape(label_v,(1,length))
value_v = np.reshape(value_v,(1,length))
diff_v = np.reshape(diff_v,(1,length))

result = np.zeros((length,3))

result[:,0] = value_v[:,:]
result[:,1] = label_v[:,:]
result[:,2] = diff_v[:,:]

header = 'predict,label,difference'
np.savetxt('enlc_result.csv', result, delimiter=',', header=header, comments='')