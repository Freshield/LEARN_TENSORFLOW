import numpy as np
import pandas as pd

model = 233


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

# get a random data(maybe have same value)
#ver 1.0
def get_batch_data(X_dataset, para_dataset, y_dataset, batch_size):
    lines_num = X_dataset.shape[0]
    random_index = np.random.randint(lines_num, size=[batch_size])
    X_data = X_dataset[random_index]
    para_data = para_dataset[random_index]
    y_data = y_dataset[random_index]
    return {'X': X_data, 'p':para_data, 'y': y_data}

# directly get whole dataset(only for small dataset)
#ver 1.0
def get_whole_data(X_dataset, para_dataset, y_dataset):
    return {'X': X_dataset, 'p':para_dataset, 'y': y_dataset}

#get a random indexs for dataset,
#so that we can shuffle the data every epoch
#ver 1.0
def get_random_seq_indexs(data_set):
    data_size = data_set.shape[0]
    #index = tf.random_shuffle(tf.range(0, data_size))#maybe can do it on tensorflow later
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    return indexs

# get a random indexs for file,
# so that we can shuffle the data every epoch
#ver 1.0
def get_file_random_seq_indexs(num):
    indexs = np.arange(num)
    np.random.shuffle(indexs)
    return indexs


# use the indexs together,
# so that we can sequence batch whole dataset
#ver 1.0
def sequence_get_data(X_dataset, para_dataset, y_dataset, indexs, last_index, batch_size):
    next_index = last_index + batch_size

    if next_index > X_dataset.shape[0]:

        next_index -= X_dataset.shape[0]
        last_part = np.arange(last_index, indexs.shape[0])
        before_part = np.arange(next_index)
        span_index = indexs[np.concatenate((last_part, before_part))]  # link two parts together
        out_of_dataset = True
    else:
        span_index = indexs[last_index:next_index]
        out_of_dataset = False

    X_data = X_dataset[span_index]
    para_data = para_dataset[span_index]
    y_data = y_dataset[span_index]
    return (next_index, {'X': X_data, 'p':para_data, 'y': y_data}, out_of_dataset)

#set num to one hot array
#ver 1.0
def num_to_one_hot(dataset, category_num):
    lines = dataset.shape[0]
    one_hot_dataset = np.zeros([lines, category_num], dtype=np.float32)
    one_hot_dataset[np.arange(lines), dataset] = 1
    return one_hot_dataset

#read the dataset from file
#needn't changed, just reply on reshape_dataset
#ver 1.0
def prepare_dataset(dir, file, SPAN):
    filename = dir + file

    dataset = pd.read_csv(filename, header=None)
    """
    #needn't the split cause the data file was splited
    test_dataset_size = int(radio * dataset.shape[0])

    cases = {
        'train':dataset.values[0:-test_dataset_size * 2],
        'validation':dataset.values[-test_dataset_size * 2:-test_dataset_size],
        'test':dataset.values[-test_dataset_size:len(dataset)]
    }

    output = cases[model]
    """
    X_data, para_data, y_data = model.reshape_dataset(dataset.values, SPAN)
    return X_data, para_data, y_data


#read the dataset from file
#needn't changed, just reply on reshape_dataset
#ver 1.0
def prepare_dataset_inclue_enlc(dir, file, SPAN):
    filename = dir + file

    dataset = pd.read_csv(filename, header=None)
    """
    #needn't the split cause the data file was splited
    test_dataset_size = int(radio * dataset.shape[0])

    cases = {
        'train':dataset.values[0:-test_dataset_size * 2],
        'validation':dataset.values[-test_dataset_size * 2:-test_dataset_size],
        'test':dataset.values[-test_dataset_size:len(dataset)]
    }

    output = cases[model]
    """
    X_data, para_data, y_data = model.reshape_dataset(dataset.values, SPAN)
    y_true = dataset.values[:, 24060 + SPAN[0]].astype(int)
    enlc_data = dataset.values[:, 24040 + SPAN[0]]
    return X_data, para_data, y_data, y_true, enlc_data