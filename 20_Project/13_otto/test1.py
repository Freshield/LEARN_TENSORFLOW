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

b = 115
c = 10
print b // c
print b % c
def mininum_values(values1, values2):
    min_value1 = values1
    min_value2 = values2

    min_values = np.minimum(min_value1,min_value2)

    return min_values

d = np.array([1,2,3,4])
e = np.array([1,1,4,3])

print mininum_values(d,e)

f = int(0.1 * 1878)
print f