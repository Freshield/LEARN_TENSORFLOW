import numpy as np

a = np.zeros((4,5))

print a

b = np.arange(4)
b += 1

print b

a[:,0] = b
print a