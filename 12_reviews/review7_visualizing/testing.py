import numpy as np

a = [x for x in range(10)]
print a
a = np.arange(12)
print a
b = a.reshape([3, 4])
print b
c = np.arange(5, 17, 1).reshape([4, 3])
print c

print np.resize(c, [5, 5])