import numpy as np
import pandas as pd

print np.random.randint(10, size=[20])

#dataset = pd.read_csv('ciena_test.csv', header=none)
#print dataset.shape

a = np.arange(12,dtype=np.float32).reshape([3,4])

print a

print np.mean(a[:,:-1])

a[:,:-1] -= np.mean(a[:,:-1])

print a

regs = np.random.uniform(-5, -1)

print regs