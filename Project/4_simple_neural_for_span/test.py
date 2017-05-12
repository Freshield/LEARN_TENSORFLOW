import numpy as np
import pandas as pd

print np.random.randint(10, size=[20])

dataset = pd.read_csv('ciena_test.csv', header=None)
print dataset.shape