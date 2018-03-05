import pandas as pd
import numpy as np

a = np.arange(10).reshape((2,5))

b = np.ones((5,1)) + 3

c = np.matmul(a,b)

print a
print b
print c