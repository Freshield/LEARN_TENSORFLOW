import tensorflow as tf
import numpy as np
import pandas as pd

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
