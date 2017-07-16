import numpy as np

a = np.arange(0.0,-10,-1.0)
print a

c = []
for i in range(10):
    c.append('%s'%a[i])

print c

b = 0.49

if b > a.min():
    index = a.argmin()
    a[index] = b
    c[index] = '%s'%b

print a
print c