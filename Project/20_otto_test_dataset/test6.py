import numpy as np

a = np.arange(20,dtype=np.float32).reshape((4,5))

print a

b = (np.arange(4) + 1).reshape((4,1))

print b

c = np.concatenate((b,a), axis=1)

print c
