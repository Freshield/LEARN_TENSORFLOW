import numpy as np

a = np.ones([3,4])

print a

lines_num = a.shape[0]
print lines_num
random_index = np.random.randint(lines_num, size=[5])
print random_index