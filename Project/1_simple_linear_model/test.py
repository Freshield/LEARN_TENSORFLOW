import numpy as np
import pandas as pd
import tensorflow as tf
import os
print 100 % 1000
print 1100 % 1000
"""
a = np.array([0,1,2,3,4,5,6,7,8,9])

print len(a)

def get(data, last_index, batch):
    next_index = last_index + batch
    if next_index > len(data):
        last_index -= len(data)
        next_index -= len(data)
    indexs = np.arange(last_index, next_index, 1)
    print indexs
    mean = np.mean(data[indexs])
    return next_index, mean

b = get(a, 10, 4)
print b

print 104 / 10
print 104 % 10

def do_eval(data_set, batch_size):
    num_epoch = len(data_set) / batch_size
    reset_data_size = len(data_set) % batch_size

    index = 0
    count = 0
    for step in xrange(num_epoch):
        index, mean = get(data_set, index, batch_size)
        count += mean * batch_size
    if reset_data_size != 0:
        #the reset data
        index, mean = get(data_set, index, reset_data_size)
        count += mean * reset_data_size
    return float(count) / len(data_set)

x = np.arange(100)
print np.mean(x)

print do_eval(x, 20)

filename = '/home/freshield/ciena_test/FiberID_Data.csv'
dataset = pd.read_csv(filename, header=None)

print dataset.shape
print dataset.values[1,:]
"""
for x in [1,2]:
    print x

print '%s' % True

"""
filename = 'modules/test'
f = file(filename, 'w+')
f.write("1")
f.close()


def copyFiles(sourceDir,  targetDir):
    if sourceDir.find(".csv") > 0:
        print 'error'
        return
    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir,  file)
        targetFile = os.path.join(targetDir,  file)
        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or(os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))):
                    open(targetFile, "wb").write(open(sourceFile, "rb").read())
        if os.path.isdir(sourceFile):
            First_Directory = False
            copyFiles(sourceFile, targetFile)

path = 'tmp'
if tf.gfile.Exists(path):
    tf.gfile.DeleteRecursively(path)
tf.gfile.MakeDirs(path)
copyFiles('modules', 'tmp')

def sequence_get_data(data_set, indexs, last_index, batch_size):
    next_index = last_index + batch_size
    out_of_dataset = False

    if next_index > data_set.shape[0]:

        next_index -= data_set.shape[0]
        last_part = np.arange(last_index,indexs.shape[0])
        before_part = np.arange(next_index)
        span_index = indexs[np.concatenate((last_part, before_part))]
        out_of_dataset = True
    else:
        span_index = indexs[last_index:next_index]


    columns = data_set[span_index]

    return (next_index, columns, out_of_dataset)
a = np.array([0,1,2,3,4])
indexs = np.array([0,1,2,3,4])

last_index = 0
for i in range(10):
    last_index, columns, out_of_dataset = sequence_get_data(a, indexs, last_index, 2)

    print last_index
    print columns
    print out_of_dataset


def write_file(result, dir_path, situation_now):
    filename = 'modules/%f-%s' % (result, dir_path)
    f = file(filename, 'w+')
    f.write(dir_path)
    f.write(situation_now)
    f.close()
    print 'best file writed'

write_file(15, 'abc', '\nhello\n')

dataset = pd.read_csv('ciena_test.csv', header=None)
print dataset.shape
"""
def test():
    for i in range(20):
        print i
        if i == 10:
            break

test()