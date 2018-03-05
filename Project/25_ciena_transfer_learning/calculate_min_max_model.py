import numpy as np
import pandas as pd
import time




#calculate the min max value first












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

def split_indexs(indexs, test_dataset_size=None, radio=None):
    if radio != None:
        test_dataset_size = int(radio * len(indexs))

    train_set = indexs[0:-test_dataset_size * 2]
    validation_set = indexs[-test_dataset_size * 2:-test_dataset_size]
    test_set = indexs[-test_dataset_size:len(indexs)]

    return train_set, validation_set, test_set

#create total mininum and total maxium values
#ver 1.0
def create_min_max():
    CMr_min = 0.0
    CMi_min = 0.0
    CD_min = 0.0
    length_min = 0.0
    power_min = 0.0

    CMr_max = 0.0
    CMi_max = 0.0
    CD_max = 0.0
    length_max = 0.0
    power_max = 0.0

    min_values = (CMr_min, CMi_min, CD_min, length_min, power_min)
    max_values = (CMr_max, CMi_max, CD_max, length_max, power_max)

    return min_values, max_values

#get the min and max values in dataset
#ver 1.0
def get_min_max_values(norm_dataset):
    CMr_min = np.min(norm_dataset[:, 0:12000])
    CMi_min = np.min(norm_dataset[:, 12000:24000])
    CD_min = np.min(norm_dataset[:, 24000:24001])
    length_min = np.min(norm_dataset[:, 24001:24021])
    power_min = np.min(norm_dataset[:, 24021:24041])

    CMr_max = np.max(norm_dataset[:, 0:12000])
    CMi_max = np.max(norm_dataset[:, 12000:24000])
    CD_max = np.max(norm_dataset[:, 24000:24001])
    length_max = np.max(norm_dataset[:, 24001:24021])
    power_max = np.max(norm_dataset[:, 24021:24041])

    min_values = (CMr_min, CMi_min, CD_min, length_min, power_min)
    max_values = (CMr_max, CMi_max, CD_max, length_max, power_max)

    return min_values, max_values

#add the values into array for save in csv file
#ver 1.0
def add_values_to_array(values, array, num):
    CMr, CMi, CD, length, power = values

    array[num,0] = CMr
    array[num,1] = CMi
    array[num,2] = CD
    array[num,3] = length
    array[num,4] = power

#compare and return the mininum values
#ver 1.0
def mininum_values(values1, values2):
    CMr_1, CMi_1, CD_1, length_1, power_1 = values1
    CMr_2, CMi_2, CD_2, length_2, power_2 = values2

    CMr = min(CMr_1, CMr_2)
    CMi = min(CMi_1, CMi_2)
    CD = min(CD_1, CD_2)
    length = min(length_1, length_2)
    power = min(power_1, power_2)

    output = (CMr, CMi, CD, length, power)

    return output

#compare and return the maxium values
#ver 1.0
def maxium_values(values1, values2):
    CMr_1, CMi_1, CD_1, length_1, power_1 = values1
    CMr_2, CMi_2, CD_2, length_2, power_2 = values2

    CMr = max(CMr_1, CMr_2)
    CMi = max(CMi_1, CMi_2)
    CD = max(CD_1, CD_2)
    length = max(length_1, length_2)
    power = max(power_1, power_2)

    output = CMr, CMi, CD, length, power

    return output

#calculate and restore min and max
#ver 1.0
def cal_minmax_split_single_file(filename, save_path):
    #filename = '/media/freshield/LINUX/Ciena/CIENA/raw/FiberID_Data_noPCA.csv'
    #savename = "/media/freshield/LINUX/Ciena/CIENA/raw/min_max.csv"

    data = pd.read_csv(filename, header=None, dtype=np.float32)

    indexs = np.arange(len(data))
    np.random.shuffle(indexs)

    train_index, validation_index, test_index = split_indexs(indexs, radio=0.1)

    train_set = data.values[train_index,:]
    validation_set = data.values[validation_index,:]
    test_set = data.values[test_index,:]

    # get min and max
    min_values, max_values = get_min_max_values(train_set)

    #split the dataset name
    savename = filename.split('/')[-1].split('.')[0]

    np.savetxt(save_path+savename+'_train.csv', train_set, delimiter=',')
    np.savetxt(save_path+savename+'_valid.csv', validation_set, delimiter=',')
    np.savetxt(save_path+savename+'_test.csv', test_set, delimiter=',')

    return min_values,max_values

def cal_minmax_split_files(dir_path,temp_name,file_amount,save_path):
    print 'Begin to split files'

    total_min, total_max = create_min_max()

    for filenum in range(file_amount):

        print filenum

        before_time = time.time()

        filename = dir_path + temp_name + str(filenum) + '.csv'

        minVal, maxVal = cal_minmax_split_single_file(filename,save_path)

        if filenum == 0:
            total_min = minVal
            total_max = maxVal
        else:
            # compare values
            total_min = mininum_values(total_min, minVal)
            total_max = maxium_values(total_max, maxVal)

        if filenum % 50 == 0:
            span_time = time.time() - before_time
            print "use %.2f second in 10 loop" % (span_time * 10)
            print "need %.2f minutes for all loop" % (((file_amount - filenum) * span_time) / 60)

    array = np.zeros([2, 5], dtype=np.float32)
    add_values_to_array(total_min, array, 0)
    add_values_to_array(total_max, array, 1)
    np.savetxt(save_path+'minmax_value.csv', array, delimiter=",")


def cal_minmax_raw_file(filename):

    reader = pd.read_csv(filename, header=None, iterator=True, dtype=np.float32)

    chunkSize = 1000

    loop = True

    count = 0

    total_min, total_max = create_min_max()

    while loop:
        try:
            #get the chunk
            chunk = reader.get_chunk(chunkSize).values
            #get the min max values
            minVal, maxVal = get_min_max_values(chunk)

            if count == 0:
                total_min = minVal
                total_max = maxVal
            else:
                # compare values
                total_min = mininum_values(total_min, minVal)
                total_max = maxium_values(total_max, maxVal)

            print count

            count += 1


        except StopIteration:
            print "stop"
            break

    return total_min, total_max

def cal_minmax_raw_files(dir_path, file_name, file_amount):
    raw_name = dir_path + file_name

    total_min, total_max = create_min_max()

    for file_num in xrange(file_amount):
        filename = raw_name + str(file_num+1) + '.csv'

        print 'Now to process the file: ' + file_name + str(file_num+1) + '.csv'

        minVal, maxVal = cal_minmax_raw_file(filename)

        if file_num == 0:
            total_min = minVal
            total_max = maxVal
        else:
            # compare values
            total_min = mininum_values(total_min, minVal)
            total_max = maxium_values(total_max, maxVal)

    print total_min
    print total_max
    array = np.zeros([2, 5], dtype=np.float32)
    add_values_to_array(total_min, array, 0)
    add_values_to_array(total_max, array, 1)
    np.savetxt('minmax_value.csv', array, delimiter=",")


dir_path = '/media/freshield/New_2T_Data/Ciena/new_data/FiberID_6fibers_20Spans/'

file_name = 'FiberID_6fibers_20Spans_noPCA_'

#cal_minmax_raw_files(dir_path, file_name, 10)



#cal_minmax_split_files('/media/freshield/DATA_W/Ciena_new_data/10spans/','Raw_data_',1000,'/media/freshield/DATA_W/Ciena_new_data/10spans_split/')
#minVal,maxVal = cal_minmax_split_single_file('sample/sample_set.csv','sample/')
#array = np.zeros([2, 5], dtype=np.float32)
#add_values_to_array(minVal, array, 0)
#add_values_to_array(maxVal, array, 1)
#np.savetxt('sample/minmax_value.csv', array, delimiter=",")
#cal_min_max('/home/freshield/Ciena_data/dataset_10k/ciena10000.csv','/home/freshield/Ciena_data/dataset_10k/model/min_max.csv',10000, 100)
"""
dir_path = '/media/freshield/DATA/Ciena_new_data/20spans/split/'
temp_name = 'Raw_data_'
file_amount = 1000
save_path = '/media/freshield/DATA/Ciena_new_data/20spans/minmax/'


cal_minmax_split_files(dir_path,temp_name,file_amount,save_path)
"""
