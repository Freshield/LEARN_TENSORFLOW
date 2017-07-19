import numpy as np

a = np.array([1,2,1,1])

b = np.arange(8).reshape((2,4))

print b

def get_value_index(number, array):
    index = []
    for i in range(len(array)):
        if array[i] == number:
            index.append(i)
    return index

index = get_value_index(1,a)

print b[:,index]