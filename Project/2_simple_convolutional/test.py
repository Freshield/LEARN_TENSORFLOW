import numpy as np

up = np.zeros([1, 3])
mid = np.ones([2, 3])
down = np.zeros([1, 3])

print up
print mid
print down

up_mid = np.row_stack((up, mid))
print up_mid