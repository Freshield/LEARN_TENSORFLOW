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
    columns = data_set.values[random_index]
    real_C = columns[:, :3100]
    imag_C = columns[:, 3100 : 6200]
    others = columns[:, 6200 : 6241]
    labels = columns[:, -1]

    return {'real_C':real_C, 'imag_C':imag_C, 'others':others, 'labels':labels}


def get_whole_data(data_set):
    columns = data_set.values
    real_C = columns[:, :3100]
    imag_C = columns[:, 3100: 6200]
    others = columns[:, 6200: 6241]
    labels = columns[:, -1]

    return {'real_C': real_C, 'imag_C': imag_C, 'others': others, 'labels': labels}

def sequence_get_data(data_set, last_index, batch_size):
    next_index = last_index + batch_size
    if next_index > len(data_set):
        last_index -= len(data_set)
        next_index -= len(data_set)
    indexs = np.arange(last_index, next_index, 1)

    columns = data_set.values[indexs]
    real_C = columns[:, :3100]
    imag_C = columns[:, 3100: 6200]
    others = columns[:, 6200: 6241]
    labels = columns[:, -1]

    return {'real_C': real_C, 'imag_C': imag_C, 'others': others, 'labels': labels}

############################################################
############### test #######################################
############################################################

filename = 'ciena1000.csv'

dataset = pd.read_csv(filename, header=None)

print dataset.values.shape

train_dataset, validation_dataset, test_dataset = split_dataset(dataset, radio=0.1)

data = get_batch_data(train_dataset, 100)



with tf.Graph().as_default():
    with tf.Session() as sess:
        #inputs
        real_C_pl = tf.placeholder(tf.float32, [None, 3100])
        imag_C_pl = tf.placeholder(tf.float32, [None, 3100])
        labels_pl = tf.placeholder(tf.int32, [None])

        #reshape
        real_C_reshape = tf.reshape(real_C_pl, [-1, 31, 100])
        imag_C_reshape = tf.reshape(imag_C_pl, [-1, 31, 100])
        image_no_padding = tf.concat([real_C_reshape, imag_C_reshape], axis=1)

        #placeholders
        images_pl = tf.pad(image_no_padding, [[0,0], [3,3], [0,0]], 'CONSTANT')
        labels_one_hot_pl = tf.one_hot(labels_pl, 3)
        others_pl = tf.placeholder(tf.float32, [None, 41])




