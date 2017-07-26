import numpy as np

a = np.arange(20).reshape((4,5)).astype(np.float32)

print a

min = np.min(a, axis=0)
max = np.max(a, axis=0)

print min
print max

max_mins_min = max - ((max - min) / 2)

print max_mins_min

print 'a - max_mins_min'

print a - max_mins_min

norm = (a - max_mins_min) / (max - max_mins_min)

print norm

norm2 = (2 * a - max - min) / (max - min)

print norm2