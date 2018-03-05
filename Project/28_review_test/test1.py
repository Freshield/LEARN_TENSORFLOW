import numpy as np

f1 = 1
f2 = 2
f3 = 3
f4 = 4
print (f1,f2)

f = []
f[0:0] = (f1,f2)
print f

f[0:0] = [f3,f4]
print f
