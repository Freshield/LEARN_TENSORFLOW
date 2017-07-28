import numpy as np
import pandas as pd
import time


# split the dataset into three part:
# training, validation, test
#ver 1.0
def split_dataset(dataset, test_dataset_size=None, radio=None):
    if radio != None:
        test_dataset_size = int(radio * len(dataset))

    train_set = dataset.values[0:-test_dataset_size * 2]
    validation_set = dataset.values[-test_dataset_size * 2:-test_dataset_size]
    test_set = dataset.values[-test_dataset_size:len(dataset)]

    return train_set, validation_set, test_set

#create total mininum and total maxium values
#ver 1.0
def create_min_max(size):

    min_values = np.zeros(size, dtype=np.float32)
    max_values = np.zeros(size, dtype=np.float32)

    return min_values, max_values

#get the min and max values in dataset
#ver 1.0
def get_min_max_values(norm_dataset):
    print norm_dataset.shape
    values = norm_dataset[:,1:-1]
    min_values = np.min(values, axis=0)
    max_values = np.max(values, axis=0)

    return min_values, max_values

#compare and return the mininum values
#ver 1.0
def mininum_values(values1, values2):

    min_values = np.minimum(values1,values2)

    return min_values

#compare and return the maxium values
#ver 1.0
def maxium_values(values1, values2):

    max_values = np.maximum(values1,values2)

    return max_values

#calculate and restore min and max
#ver 1.0
def cal_min_max(filename, savename, datasize, chunkSize):
    #filename = '/media/freshield/LINUX/Ciena/CIENA/raw/FiberID_Data_noPCA.csv'
    #savename = "/media/freshield/LINUX/Ciena/CIENA/raw/min_max.csv"

    reader = pd.read_csv(filename, iterator=True)

    loop = True
    #chunkSize = 10000
    count = 0

    total_loop = datasize // chunkSize

    total_min, total_max = create_min_max(93)

    print 'begin to calculate the min max value'
    while loop:
        before_time = time.time()
        try:

            print count
            chunk = reader.get_chunk(chunkSize)

            train_set, validation_set, test_set = split_dataset(chunk, radio=0.1)

            # get min and max
            min_values, max_values = get_min_max_values(train_set)

            if count == 0:
                total_min = min_values
                total_max = max_values
            else:
                # compare values
                total_min = mininum_values(total_min, min_values)
                total_max = maxium_values(total_max, max_values)

            if count % 10 == 0:
                span_time = time.time() - before_time
                print "use %.2f second in 10 loop" % (span_time * 10)
                print "need %.2f minutes for all loop" % (((total_loop - count) * span_time) / 60)

            # i += chunk.shape[0]
            count += 1


        except StopIteration:
            print "stop"
            break

    print total_max
    print total_min

    array = np.zeros([2, 93])
    array[0] = total_min
    array[1] = total_max
    np.savetxt(savename, array, delimiter=",")

#cal_min_max('data/train.csv','data/min_max.csv',61878,10000)

