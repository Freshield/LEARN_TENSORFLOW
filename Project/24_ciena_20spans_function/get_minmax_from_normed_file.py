import pandas as pd
import numpy as np

def get_min_max_values(norm_dataset, filenum, loopnum):
    CMr_min = np.min(norm_dataset[:, 0:12000])
    CMi_min = np.min(norm_dataset[:, 12000:24000])
    CD_min = np.min(norm_dataset[:, 24000:24001])
    length_min = np.min(norm_dataset[:, 24001:24021])
    power_min = np.min(norm_dataset[:, 24021:24041])

    CMr_min_pos = (np.where(norm_dataset[:, 0:12000] == CMr_min),filenum,loopnum)
    CMi_min_pos = (np.where(norm_dataset[:, 12000:24000] == CMi_min),filenum,loopnum)
    CD_min_pos = (np.where(norm_dataset[:, 24000:24001] == CD_min),filenum,loopnum)
    length_min_pos = (np.where(norm_dataset[:, 24001:24021] == length_min),filenum,loopnum)
    power_min_pos = (np.where(norm_dataset[:, 24021:24041] == power_min),filenum,loopnum)

    CMr_max = np.max(norm_dataset[:, 0:12000])
    CMi_max = np.max(norm_dataset[:, 12000:24000])
    CD_max = np.max(norm_dataset[:, 24000:24001])
    length_max = np.max(norm_dataset[:, 24001:24021])
    power_max = np.max(norm_dataset[:, 24021:24041])

    CMr_max_pos = (np.where(norm_dataset[:, 0:12000] == CMr_max),filenum,loopnum)
    CMi_max_pos = (np.where(norm_dataset[:, 12000:24000] == CMi_max),filenum,loopnum)
    CD_max_pos = (np.where(norm_dataset[:, 24000:24001] == CD_max),filenum,loopnum)
    length_max_pos = (np.where(norm_dataset[:, 24001:24021] == length_max),filenum,loopnum)
    power_max_pos = (np.where(norm_dataset[:, 24021:24041] == power_max),filenum,loopnum)

    min_values = (CMr_min, CMi_min, CD_min, length_min, power_min)
    max_values = (CMr_max, CMi_max, CD_max, length_max, power_max)
    min_pos = (CMr_min_pos, CMi_min_pos, CD_min_pos, length_min_pos, power_min_pos)
    max_pos = (CMr_max_pos, CMi_max_pos, CD_max_pos, length_max_pos, power_max_pos)

    return min_values, max_values, min_pos, max_pos

def mininum_values(min_values1, min_values2, min_pos1, min_pos2):

    def get_min(value1, value2, pos1, pos2):
        if value1 < value2:
            min_value = value1
            min_pos = pos1
        else:
            min_value = value2
            min_pos = pos2

        return min_value, min_pos


    CMr_1, CMi_1, CD_1, length_1, power_1 = min_values1
    CMr_2, CMi_2, CD_2, length_2, power_2 = min_values2
    CMr_p1, CMi_p1, CD_p1, length_p1, power_p1 = min_pos1
    CMr_p2, CMi_p2, CD_p2, length_p2, power_p2 = min_pos2

    CMr, CMr_p = get_min(CMr_1, CMr_2, CMr_p1, CMr_p2)
    CMi, CMi_p = get_min(CMi_1, CMi_2, CMi_p1, CMi_p2)
    CD, CD_p = get_min(CD_1, CD_2, CD_p1, CD_p2)
    length, length_p = get_min(length_1, length_2, length_p1, length_p2)
    power, power_p = get_min(power_1, power_2, power_p1, power_p2)

    output = (CMr, CMi, CD, length, power)
    output_p = (CMr_p, CMi_p, CD_p, length_p, power_p)

    return output, output_p


def maxinum_values(max_values1, max_values2, max_pos1, max_pos2):

    def get_max(value1, value2, pos1, pos2):
        if value1 > value2:
            max_value = value1
            max_pos = pos1
        else:
            max_value = value2
            max_pos = pos2

        return max_value, max_pos


    CMr_1, CMi_1, CD_1, length_1, power_1 = max_values1
    CMr_2, CMi_2, CD_2, length_2, power_2 = max_values2
    CMr_p1, CMi_p1, CD_p1, length_p1, power_p1 = max_pos1
    CMr_p2, CMi_p2, CD_p2, length_p2, power_p2 = max_pos2

    CMr, CMr_p = get_max(CMr_1, CMr_2, CMr_p1, CMr_p2)
    CMi, CMi_p = get_max(CMi_1, CMi_2, CMi_p1, CMi_p2)
    CD, CD_p = get_max(CD_1, CD_2, CD_p1, CD_p2)
    length, length_p = get_max(length_1, length_2, length_p1, length_p2)
    power, power_p = get_max(power_1, power_2, power_p1, power_p2)

    output = (CMr, CMi, CD, length, power)
    output_p = (CMr_p, CMi_p, CD_p, length_p, power_p)

    return output, output_p

def cal_minmax_single_file(filename, chunkSize, filenum):

    reader = pd.read_csv(filename, header=None, iterator=True, dtype=np.float32)

    loop = True

    count = 0

    total_min = None
    total_max = None
    total_min_p = None
    total_max_p = None

    while loop:
        try:
            #get the chunk
            chunk = reader.get_chunk(chunkSize).values
            #get the min max values
            min_values, max_values, min_pos, max_pos = get_min_max_values(chunk, filenum, count)

            if count == 0:
                total_min = min_values
                total_max = max_values
                total_min_p = min_pos
                total_max_p = max_pos
            else:
                # compare values
                total_min, total_min_p = mininum_values(total_min, min_values, total_min_p, min_pos)
                total_max, total_max_p = maxinum_values(total_max, max_values, total_max_p, max_pos)

            print count

            count += 1


        except StopIteration:
            print "stop"
            break

    return total_min, total_max, total_min_p, total_max_p

#filename1 = '/media/freshield/Passort_2T_Data_W/Ciena_new_data/20spans/norm/Raw_data_0_test.csv'

dir_name = '/media/freshield/Passort_2T_Data_W/Ciena_new_data/20spans/norm/'

total_min = None
total_min_p = None
total_max = None
total_max_p = None

file_amount = 1000
batch_size = 100

for filenum in xrange(file_amount):

    train_file_name = 'Raw_data_%d_train.csv' % filenum
    test_file_name = 'Raw_data_%d_test.csv' % filenum
    valid_file_name = 'Raw_data_%d_valid.csv' % filenum
    
    #for train
    print 'Begin for train file ' + train_file_name
    filename = dir_name + train_file_name
    min_values, max_values, min_pos, max_pos = cal_minmax_single_file(filename, batch_size, train_file_name)

    if filenum == 0:
        total_min = min_values
        total_max = max_values
        total_min_p = min_pos
        total_max_p = max_pos
    else:
        # compare values
        total_min, total_min_p = mininum_values(total_min, min_values, total_min_p, min_pos)
        total_max, total_max_p = maxinum_values(total_max, max_values, total_max_p, max_pos)

    #for test
    print 'Begin for test file ' + test_file_name
    filename = dir_name + test_file_name
    min_values, max_values, min_pos, max_pos = cal_minmax_single_file(filename, batch_size, test_file_name)
    # compare values
    total_min, total_min_p = mininum_values(total_min, min_values, total_min_p, min_pos)
    total_max, total_max_p = maxinum_values(total_max, max_values, total_max_p, max_pos)
    
    #for valid
    print 'Begin for valid file ' + valid_file_name
    filename = dir_name + valid_file_name
    min_values, max_values, min_pos, max_pos = cal_minmax_single_file(filename, batch_size, valid_file_name)
    # compare values
    total_min, total_min_p = mininum_values(total_min, min_values, total_min_p, min_pos)
    total_max, total_max_p = maxinum_values(total_max, max_values, total_max_p, max_pos)

print total_min
print total_max
print total_min_p
print total_max_p

minmax_normed = open('minmax_normed.txt', 'w')
minmax_normed.write(str(total_min) + '\n')
minmax_normed.write(str(total_max) + '\n')
minmax_normed.write(str(total_min_p) + '\n')
minmax_normed.write(str(total_max_p) + '\n')
minmax_normed.close()

"""

filename = '/media/freshield/Passort_2T_Data_U/Ciena_data/new_data/FiberID_6fibers_20Spans/FiberID_6fibers_20Spans_noPCA_2.csv'

#data = pd.read_csv(filename1, header=None).values[:50]
data = pd.read_csv(filename, header=None, nrows=50).values

min_values1, max_values1, min_pos1, max_pos1 = get_min_max_values(data[:25],'lol',0)

print str(min_values1)
print min_pos1

myfile = open('myfile.txt', 'w')
myfile.write(str(min_values1) + '\n')
myfile.write(str(min_pos1))
myfile.close()
"""