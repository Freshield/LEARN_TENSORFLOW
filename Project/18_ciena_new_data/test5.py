import numpy as np

a = np.arange(40).reshape((2,4,5))

print a

flap_a = a[:,::-1]
print
print a
print flap_a
