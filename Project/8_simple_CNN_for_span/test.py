import numpy as np

a = np.ones([3,4])

print a

lines_num = a.shape[0]
print lines_num
random_index = np.random.randint(lines_num, size=[5])
print random_index

b = np.random.randint(lines_num, size=[6])
print b

def num_to_one_hot(dataset, category_num):
    lines = dataset.shape[0]
    one_hot_dataset = np.zeros([lines,category_num], dtype=np.float32)
    one_hot_dataset[np.arange(lines), dataset] = 1
    return one_hot_dataset

print num_to_one_hot(b, 3)

c = []
c.append(10)
c.append(20)
c.append(30)

print c

c = np.arange(20)
d = c[2:12]
print len(d)

e = 10
f = 5
g = 15
h = (e,f,g)
print h

for para in h:
    print para
