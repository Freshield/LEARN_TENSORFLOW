import numpy as np

a = np.arange(20)

cases = {
    '1': a[:2],
    '2': a[2:5],
    '3': a[5:15],
    '4': a[15:]
}

print cases['3']