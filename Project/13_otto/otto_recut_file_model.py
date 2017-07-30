import numpy as np
import pandas as pd
import time

dic = {'Class_1':0, 'Class_2':1, 'Class_3':2, 'Class_4':3, 'Class_5':4, 'Class_6':5, 'Class_7':6, 'Class_8':7, 'Class_9':8}

#split the dataset into three part:
#training, validation, test
#ver 1.0
def split_dataset(dataset, test_dataset_size=None, radio=None):
    if radio != None:
        test_dataset_size = int(radio * len(dataset))

    indexs = np.arange(len(dataset))
    np.random.shuffle(indexs)

    train_set = dataset.values[indexs[0:-test_dataset_size * 2]]
    validation_set = dataset.values[indexs[-test_dataset_size * 2:-test_dataset_size]]
    test_set = dataset.values[indexs[-test_dataset_size:len(dataset)]]

    return train_set, validation_set, test_set

#normalize the dataset, push data between -1 to 1
#ver 1.0
def normalize_dataset(dataset, min_values=None, max_values=None, type='train'):

    def calcul_norm(dataset, min, max):
        return (2 * dataset - max - min) / (max - min)

    if type == 'train':
        x,y = dataset.shape
        norm_dataset = np.zeros((x,y-1))
        values = np.zeros((x,y-2))
        values[:,:] = dataset[:,1:-1]
        for i in range(len(norm_dataset)):
            norm_dataset[i,-1] = dic[dataset[i,-1]]
    elif type == 'test':
        x, y = dataset.shape
        norm_dataset = np.zeros((x, y - 1))
        values = np.zeros((x, y - 1))
        values[:,:] = dataset[:, 1:]

    if min_values == None:
        min_values = np.min(values, axis=0)

    if max_values == None:
        max_values = np.max(values, axis=0)

    values = calcul_norm(values, min_values, max_values)

    if type == 'train':
        norm_dataset[:,:-1] = values[:,:]
    elif type == 'test':
        norm_dataset[:,:] = values[:,:]

    return norm_dataset

#recut the normalize and split the dataset
#ver 1.0
def norm_recut_dataset(filename, savePath, minmax_name, dataSize, chunkSize, type='train'):
    #filename = '/media/freshield/LINUX/Ciena/CIENA/raw/FiberID_Data_noPCA.csv'
    #chunkSize = 1000
    #savePath = '/media/freshield/LINUX/Ciena/CIENA/raw/norm/'
    #minmax_name = '/media/freshield/LINUX/Ciena/CIENA/raw/min_max.csv'

    reader = pd.read_csv(filename, iterator=True)

    loop = True
    # to do
    # think about chunkSize 10000

    total_loop = dataSize // chunkSize

    i = 0
    count = 0

    minmax_array = pd.read_csv(minmax_name, header=None, dtype=np.float32).values

    # get min and max
    min_values = minmax_array[0]
    max_values = minmax_array[1]

    print 'begin to norm and recut the file'

    while loop:
        before_time = time.time()
        try:
            chunk = reader.get_chunk(chunkSize)

            if type == 'train':
                train_set, validation_set, test_set = split_dataset(chunk, radio=0.025)

                # normalize dataset
                train_set = normalize_dataset(train_set, min_values, max_values, type)
                validation_set = normalize_dataset(validation_set, min_values, max_values, type)
                test_set = normalize_dataset(test_set, min_values, max_values, type)

                np.savetxt(savePath + "train_set_%d.csv" % count, train_set, delimiter=",")
                np.savetxt(savePath + "validation_set_%d.csv" % count, validation_set, delimiter=",")
                np.savetxt(savePath + "test_set_%d.csv" % count, test_set, delimiter=",")
            elif type == 'test':
                test_set = chunk.values
                test_set = normalize_dataset(test_set, min_values, max_values, type)
                np.savetxt(savePath + "test_set_%d.csv" % count, test_set, delimiter=",")

            if count % 10 == 0:
                span_time = time.time() - before_time
                print "use %.2f second in 10 loop" % (span_time * 10)
                print "need %.2f minutes for all loop" % (((total_loop - count) * span_time) / 60)

            # i += chunk.shape[0]
            count += 1
            print count



        except StopIteration:
            print "stop"
            break

#norm_recut_dataset('data/train.csv', 'data/norm/train/', 'data/min_max.csv', 61878, 10000, 'train')
#norm_recut_dataset('data/test.csv', 'data/norm/test/', 'data/min_max.csv', 144368, 10000, 'test')
