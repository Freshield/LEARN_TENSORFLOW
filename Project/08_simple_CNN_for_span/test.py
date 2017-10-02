import numpy as np
import pandas as pd
import time

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

def change_array(array):
    array[3,3] = 0

i = np.ones([24]).reshape([4,6])
print i
change_array(i)
print i

print np.mean(i, axis=0)

j = max(1,2)
print j


#filename = '/media/freshield/LINUX/Ciena/CIENA/raw/FiberID_Data_noPCA.csv'
#filename = "/media/freshield/LINUX/Ciena/CIENA/raw/norm/train_set_0.csv"

#data = pd.read_csv(filename,skiprows=20, header=None, nrows=100)

#reader = pd.read_csv(filename, header=None)#, iterator=True)
"""
try:
    df = reader.get_chunk(10)
    print df.shape
except StopIteration:
    print "Iteration is stopped."


loop = True
chunkSize = 500
i = 0
count = 0

befor_time = time.time()
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        i += chunk.shape[0]
        count += 1
        print count

    except StopIteration:
        print "stop"
        break
span_time = time.time() - befor_time
print "use %.2f second" % span_time
print i
"""
#print reader.shape