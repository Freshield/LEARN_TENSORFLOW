import numpy as np
import pandas as pd

dir_name = '/home/freshield/'
score_filename = 'span1_value_acc_0.9948.csv'
label_filename = 'span1_label_acc_0.9948.csv'

y_score = pd.read_csv(dir_name+score_filename, header=None).values
y_test = pd.read_csv(dir_name+label_filename, header=None, dtype=np.int32).values

print y_score[0:5]
print y_test[0:5]
