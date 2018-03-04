import numpy as np

def calcul_norm(dataset, min, max):
    return (2 * dataset - max - min) / (max - min)

def calcul_norm_new(dataset, min, max, mean):
    return (2 * dataset - 2 * mean) / (max - min)

data = np.arange(20, dtype=np.float32).reshape((4,5))
print data

min = np.min(data, axis=0)
max = np.max(data, axis=0)
mean = np.mean(data, axis=0)

print min
print max
print mean
print

print calcul_norm(data, min, max)
print
print calcul_norm_new(data, min, max, mean)