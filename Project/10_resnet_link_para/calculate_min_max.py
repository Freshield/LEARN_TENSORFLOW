import numpy as np
import pandas as pd


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
    CMr_min = np.min(norm_dataset[:, 0:3100])
    CMi_min = np.min(norm_dataset[:, 3100:6200])
    CD_min = np.min(norm_dataset[:, 6200:6201])
    length_min = np.min(norm_dataset[:, 6201:6221])
    power_min = np.min(norm_dataset[:, 6221:6241])

    CMr_max = np.max(norm_dataset[:, 0:3100])
    CMi_max = np.max(norm_dataset[:, 3100:6200])
    CD_max = np.max(norm_dataset[:, 6200:6201])
    length_max = np.max(norm_dataset[:, 6201:6221])
    power_max = np.max(norm_dataset[:, 6221:6241])

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

######################################################################
######################### Main function ##############################
######################################################################

filename = '/media/freshield/LINUX/Ciena/CIENA/raw/FiberID_Data_noPCA.csv'

reader = pd.read_csv(filename, header=None, iterator=True, dtype=np.float32)

loop = True
chunkSize = 1000
i = 0
count = 0

total_min, total_max = create_min_max()

while loop:
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
            #compare values
            total_min = mininum_values(total_min, min_values)
            total_max = maxium_values(total_max, max_values)

        #i += chunk.shape[0]
        count += 1
        #if count == 10:
        #    break

    except StopIteration:
        print "stop"
        break
#span_time = time.time() - before_time
#print "use %.2f second in 10 loop" % span_time
#print "need %.2f minutes for all 600 loop" % (span_time * 60 / 60)
#print i
print total_max
print total_min

array = np.zeros([2,5], dtype=np.float32)
add_values_to_array(total_min, array, 0)
add_values_to_array(total_max, array, 1)
np.savetxt("/media/freshield/LINUX/Ciena/CIENA/raw/min_max.csv", array, delimiter=",")

