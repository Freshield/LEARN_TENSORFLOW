import numpy as np
import tensorflow as tf

a = np.arange(20)

np.random.shuffle(a)

print a

print a.min()
print a.argmin()
print a[a.argmin()]

b = np.arange(-10,-1,1)
b[0] = -2
print b
print b.argmin()

for i in range(20):
    if a[i] > b.min():
        b[b.argmin()] = a[i]

print b

def del_dir(dir_path):
    if tf.gfile.Exists(dir_path):
        tf.gfile.DeleteRecursively(dir_path)
        return 'Done'
    else:
        return 'Error, do not have the dir path'

print del_dir('-8.0')

c = 10
d = np.arange(0,-c,-1)
print d

d = []
for i in range(c):
    d.append('%s'%i)

print d

