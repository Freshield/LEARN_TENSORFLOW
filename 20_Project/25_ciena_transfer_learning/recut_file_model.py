import numpy as np
import pandas as pd
import time

#split the dataset into three part:
#training, validation, test
#ver 1.0
def split_dataset(dataset, test_dataset_size=None, radio=None):
    if radio != None:
        test_dataset_size = int(radio * len(dataset))

    train_set = dataset.values[0:-test_dataset_size * 2]
    validation_set = dataset.values[-test_dataset_size * 2:-test_dataset_size]
    test_set = dataset.values[-test_dataset_size:len(dataset)]

    return train_set, validation_set, test_set

#normalize the dataset, push data between -1 to 1
#ver 1.0
def normalize_dataset(dataset, min_values=None, max_values=None):
    norm_dataset = np.zeros((dataset.shape))
    norm_dataset[:, :] = dataset[:, :]

    if min_values == None:
        CMr_min = np.min(norm_dataset[:, 0:12000])
        CMi_min = np.min(norm_dataset[:, 12000:24000])
        CD_min = np.min(norm_dataset[:, 24000:24001])
        length_min = np.min(norm_dataset[:, 24001:24021])
        power_min = np.min(norm_dataset[:, 24021:24041])
    else:
        CMr_min, CMi_min, CD_min, length_min, power_min = min_values


    if max_values == None:
        CMr_max = np.max(norm_dataset[:, 0:12000])
        CMi_max = np.max(norm_dataset[:, 12000:24000])
        CD_max = np.max(norm_dataset[:, 24000:24001])
        length_max = np.max(norm_dataset[:, 24001:24021])
        power_max = np.max(norm_dataset[:, 24021:24041])
    else:
        CMr_max, CMi_max, CD_max, length_max, power_max = max_values

    def calcul_norm(dataset, min, max):
        return (2 * dataset - max - min) / (max - min)

    norm_dataset[:, 0:12000] = calcul_norm(norm_dataset[:, 0:12000], CMr_min, CMr_max)
    norm_dataset[:, 12000:24000] = calcul_norm(norm_dataset[:, 12000:24000], CMi_min, CMi_max)
    norm_dataset[:, 24000:24001] = calcul_norm(norm_dataset[:, 24000:24001], CD_min, CD_max)
    norm_dataset[:, 24001:24021] = calcul_norm(norm_dataset[:, 24001:24021], length_min, length_max)
    norm_dataset[:, 24021:24041] = calcul_norm(norm_dataset[:, 24021:24041], power_min, power_max)

    min_values = (CMr_min, CMi_min, CD_min, length_min, power_min)
    max_values = (CMr_max, CMi_max, CD_max, length_max, power_max)

    return norm_dataset, min_values, max_values

#get the values from array read from csv file
#ver 1.0
def get_values_from_array(array, num):
    CMr = array[num,0]
    CMi = array[num,1]
    CD = array[num,2]
    length = array[num,3]
    power = array[num,4]

    values = CMr, CMi, CD, length, power

    return values

def get_minmax(minmax_name):
    minmax_array = pd.read_csv(minmax_name, header=None, dtype=np.float32).values

    # get min and max
    min_values = get_values_from_array(minmax_array, 0)
    max_values = get_values_from_array(minmax_array, 1)

    return min_values,max_values

#recut the normalize and split the dataset
#ver 1.0
def norm_recut_dataset(filename, savePath, minmax_name, dataSize, chunkSize):
    #filename = '/media/freshield/LINUX/Ciena/CIENA/raw/FiberID_Data_noPCA.csv'
    #chunkSize = 1000
    #savePath = '/media/freshield/LINUX/Ciena/CIENA/raw/norm/'
    #minmax_name = '/media/freshield/LINUX/Ciena/CIENA/raw/min_max.csv'

    reader = pd.read_csv(filename, header=None, iterator=True, dtype=np.float32)

    loop = True
    # to do
    # think about chunkSize 10000

    total_loop = dataSize // chunkSize

    i = 0
    count = 0

    min_values, max_values = get_minmax(minmax_name)

    print 'begin to norm and recut the file'

    while loop:
        before_time = time.time()
        try:
            chunk = reader.get_chunk(chunkSize)

            train_set, validation_set, test_set = split_dataset(chunk, radio=0.1)

            # normalize dataset
            train_set, _, _ = normalize_dataset(train_set, min_values, max_values)
            validation_set, _, _ = normalize_dataset(validation_set, min_values, max_values)
            test_set, _, _ = normalize_dataset(test_set, min_values, max_values)

            np.savetxt(savePath + "train_set_%d.csv" % count, train_set, delimiter=",")
            np.savetxt(savePath + "validation_set_%d.csv" % count, validation_set,
                       delimiter=",")
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

def norm_single_file(filename, savePath, minVal, maxVal):

    data = pd.read_csv(filename, header=None, dtype=np.float32)

    savename = filename.split('/')[-1]

    norm_data, _, _ = normalize_dataset(data.values, minVal, maxVal)

    np.savetxt(savePath + savename, norm_data, delimiter=",")

def norm_files(dir_path,temp_name,file_amount,save_path1, save_path2,minmax_name):
    print 'Begin to norm files'

    for filenum in range(file_amount):

        print filenum

        before_time = time.time()

        filename = dir_path + temp_name + str(filenum)

        minVal, maxVal = get_minmax(minmax_name)

        norm_single_file(filename+'_train.csv',save_path1,minVal,maxVal)
        norm_single_file(filename+'_valid.csv',save_path1,minVal,maxVal)
        norm_single_file(filename+'_test.csv',save_path1,minVal,maxVal)

        norm_single_file(filename+'_train.csv',save_path2,minVal,maxVal)
        norm_single_file(filename+'_valid.csv',save_path2,minVal,maxVal)
        norm_single_file(filename+'_test.csv',save_path2,minVal,maxVal)

        if filenum % 50 == 0:
            span_time = time.time() - before_time
            print "use %.2f second in 10 loop" % (span_time * 10)
            print "need %.2f minutes for all loop" % (((file_amount - filenum) * span_time) / 60)

#norm_files('/media/freshield/LINUX/Ciena/10spans/','Raw_data_',6,'/media/freshield/LINUX/Ciena/norm/','/media/freshield/LINUX/Ciena/10spans/minmax_value.csv')



"""
minVal, maxVal = get_minmax('sample/minmax_value.csv')
norm_single_file('sample/sample_set_train.csv','sample/norm/',minVal,maxVal)

#norm_recut_dataset('/home/freshield/Ciena_data/dataset_10k/ciena10000.csv','/home/freshield/Ciena_data/dataset_10k/model/','/home/freshield/Ciena_data/dataset_10k/model/min_max.csv',10000,100)

dir_path = '/media/freshield/DATA/Ciena_new_data/20spans/minmax/'
temp_name = 'Raw_data_'
file_amount = 1000
save_path1 = '/media/freshield/DATA/Ciena_new_data/20spans/norm/'
save_path2 = '/media/freshield/DATA_W/Ciena_new_data/20spans/norm/'
minmax_name = '/media/freshield/DATA/Ciena_new_data/20spans/minmax/minmax_value.csv'

norm_files(dir_path,temp_name,file_amount,save_path1,save_path2,minmax_name)
"""