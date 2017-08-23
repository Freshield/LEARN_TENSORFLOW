import numpy as np

a = np.ones((10,5))
b = np.ones((5,1))

mul = np.dot(a,b)
c = np.ones((10,1))

result = mul + c
print result.shape
print result