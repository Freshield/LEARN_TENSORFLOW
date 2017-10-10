import pandas as pd
import numpy as np

a = np.arange(20)

np.random.shuffle(a)

a = np.reshape(a, (4,5))

print a

print
print a[:,0:3]
print np.min(a[:,0:3])
print np.max(a[:,0:3])
x = np.where(a[:,0:3] == np.max(a[:,0:3]))
y = (x,1,1)
print y
print x
print x[0][0]
print x[1][0]
print


print np.min(a[:,3:])
print np.max(a[:,3:])
print np.argmin(a[3:],1)
print

a = np.array([[1,2,3],[4,5,6]])
print a
print a.argmax(axis=1)
