import pandas as pd
import numpy as np

def split_indexs(indexs, test_dataset_size=None, radio=None):
    if radio != None:
        test_dataset_size = int(radio * len(indexs))

    train_set = indexs[0:-test_dataset_size * 2]
    validation_set = indexs[-test_dataset_size * 2:-test_dataset_size]
    test_set = indexs[-test_dataset_size:len(indexs)]

    return train_set, validation_set, test_set


filename = '/media/freshield/DATA_W/Ciena_new_data/10spans/Raw_data_0.csv'

print filename.split('/')[-1].split('.')[0]

data = pd.read_csv(filename, header=None, dtype=np.float32)

indexs = np.arange(len(data))
np.random.shuffle(indexs)

train_index, validation_index, test_index = split_indexs(indexs, radio=0.1)

train_set = data.values[train_index, :]
validation_set = data.values[validation_index, :]
test_set = data.values[test_index, :]

print train_set.shape
print validation_set.shape
print test_set.shape

sample_set = data.values[indexs[0:10],:]
np.savetxt('sample_set.csv', sample_set, delimiter=',')