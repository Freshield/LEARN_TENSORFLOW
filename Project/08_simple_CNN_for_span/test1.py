import numpy as np

a = np.arange(20)

cases = {
    '1': a[:2],
    '2': a[2:5],
    '3': a[5:15],
    '4': a[15:]
}

print cases['3']

b = np.arange(0,2,0.1).reshape((4,5))
c = [1.0,2.0,3.0]

print b

c = b.astype(int)
print c

print c[0]

print "hello world",
print "lol",