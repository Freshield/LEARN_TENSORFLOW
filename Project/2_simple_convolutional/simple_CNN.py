import pandas as pd
import tensorflow as tf
import numpy as np
import time
import os
############################################################
############# helpers ######################################
############################################################
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

def split_dataset(dataset, test_dataset_size=None, radio=None):
    if radio != None:
        test_dataset_size = int(radio * len(dataset))

    train_set = dataset[0:-test_dataset_size * 2]
    validation_set = dataset[-test_dataset_size * 2:-test_dataset_size]
    test_set = dataset[-test_dataset_size:len(dataset)]


    return train_set, validation_set, test_set


def get_batch_data(data_set, batch_size):
    lines_num = data_set.shape[0] - 1
    random_index = np.random.randint(lines_num, size=[batch_size])
    real_C = data_set.values[random_index, :3100]
    imag_C = data_set.values[random_index, 3100 : 6200]
    others = data_set.values[random_index, 6200 : 6241]
    labels = data_set.values[random_index, -1]
    np_labels = np.array(labels, dtype=np.int32)
    labels_one_hot = np.eye(3)[np_labels]
    return {'real_C': real_C, 'imag_C':imag_C, 'others':others, 'labels': labels_one_hot}


def get_whole_data(data_set):
    features = data_set.values[:, :-20]
    labels = data_set.values[:, -1]
    np_labels = np.array(labels, dtype=np.int32)
    labels_one_hot = np.eye(3)[np_labels]
    return {'features': features, 'labels': labels_one_hot}

def sequence_get_data(data_set, last_index, batch_size):
    next_index = last_index + batch_size
    if next_index > len(data_set):
        last_index -= len(data_set)
        next_index -= len(data_set)
    indexs = np.arange(last_index, next_index, 1)

    features = data_set.values[indexs, :-20]
    labels = data_set.values[indexs, -1]
    np_labels = np.array(labels, dtype=np.int32)
    labels_one_hot = np.eye(3)[np_labels]
    return (next_index, {'features': features, 'labels': labels_one_hot})

############################################################
############### test #######################################
############################################################

filename = 'ciena1000.csv'

dataset = pd.read_csv(filename, header=None)

print dataset.values.shape

train_dataset, validation_dataset, test_dataset = split_dataset(dataset, radio=0.1)

