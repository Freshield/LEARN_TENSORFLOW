import numpy as np

a = np.arange(20).reshape((4,5))

print a
x = a.shape[0]
y = a.shape[1]
print x,y

right = a[:1]
left = a[1:]

print left
print right
print np.concatenate((left,right))

data = np.zeros((x,y,y))
print data
for i in xrange(x):
    for j in xrange(y):
        right = a[i,:j]
        left = a[i,j:]
        data[i,j] = np.concatenate((left, right))

print data