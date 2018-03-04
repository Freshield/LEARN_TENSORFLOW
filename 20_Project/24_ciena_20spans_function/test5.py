import pandas as pd
import numpy as np


min_norm = [-1.0164913, -1.0, -1.0, -1.0, -1.0]
max_norm = [1.0, 1.0, 1.0242046, 1.0, 1.0]

min_value = [-0.040546, -0.098555997, 2563.8999, -26.066, -10.398]
max_value = [0.041134998, 0.023029, 25980.0, 193.57001, 22.183001]

result_min = [-0.0398779952125, -0.098555997, 2563.8999, -26.066, -10.398]
result_max = [0.041134998, 0.023029, 25699.999976, 193.57001, 22.183001]

def get_min_value(raw, norm, max_v):

    min_v = (2 * raw - max_v - norm * max_v) / (1 - norm)
    return min_v

def get_max_value(raw, norm, min_v):

    max_v = (2 * raw - min_v + norm * min_v) / (norm + 1)
    return max_v

def calcul_norm(dataset, min, max):
    return (2 * dataset - max - min) / (max - min)

for i in range(len(min_value)):
    min_v = result_min[i]
    max_v = result_max[i]

    value1 = min_value[i]
    value2 = max_value[i]

    print calcul_norm(value1, min_v, max_v)
    print min_norm[i]
    print calcul_norm(value2, min_v, max_v)
    print max_norm[i]
    print


#print get_min_value(-0.040546, -1.0164913, 0.041134998)
#print get_max_value(25980.0, 1.0242046, 2563.8999)

