import numpy as np
import pandas as pd

score_filename = 'result_value.csv'
label_filename = 'result_label.csv'

y_score = pd.read_csv(score_filename, header=None).values
y_test = pd.read_csv(label_filename, header=None, dtype=np.int32).values

print y_score.shape
print y_test.shape
