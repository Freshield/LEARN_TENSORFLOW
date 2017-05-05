import pandas as pd
import tensorflow as tf
import numpy as np
import time
import os

################################################
#                                              #
#                 CNN models                   #
#         create CNN use those models          #
#                                              #
################################################


############################################################
############# helpers ######################################
############################################################
#to copy files from source dir to target dir
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

#split the dataset into three part:
#training, validation, test
def split_dataset(dataset, test_dataset_size=None, radio=None):
    if radio != None:
        test_dataset_size = int(radio * len(dataset))

    train_set = dataset[0:-test_dataset_size * 2]
    validation_set = dataset[-test_dataset_size * 2:-test_dataset_size]
    test_set = dataset[-test_dataset_size:len(dataset)]


    return train_set, validation_set, test_set

#get a random data(maybe have same value)
def get_batch_data(data_set, batch_size, span_num=20):
    lines_num = data_set.shape[0] - 1
    random_index = np.random.randint(lines_num, size=[batch_size])
    columns = data_set.values[random_index]
    real_C = columns[:, :3100]
    imag_C = columns[:, 3100 : 6200]
    others = columns[:, 6200 : 6241]
    label_pos = span_num - 21
    labels = columns[:, label_pos]

    return {'real_C':real_C, 'imag_C':imag_C, 'others':others, 'labels':labels}

#directly get whole dataset(only for small dataset)
def get_whole_data(data_set, span_num=20):
    columns = data_set.values
    real_C = columns[:, :3100]
    imag_C = columns[:, 3100: 6200]
    others = columns[:, 6200: 6241]
    label_pos = span_num - 21
    labels = columns[:, label_pos]

    return {'real_C': real_C, 'imag_C': imag_C, 'others': others, 'labels': labels}

#get a random indexs for dataset,
#so that we can shuffle the data every epoch
def get_random_seq_indexs(data_set):
    data_size = data_set.shape[0]
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    return indexs

#use the indexs together,
#so that we can sequence batch whole dataset
def sequence_get_data(data_set, indexs, last_index, batch_size, span_num=20):
    next_index = last_index + batch_size
    out_of_dataset = False

    if next_index > data_set.shape[0]:

        next_index -= data_set.shape[0]
        last_part = np.arange(last_index,indexs.shape[0])
        before_part = np.arange(next_index)
        span_index = indexs[np.concatenate((last_part, before_part))]
        out_of_dataset = True
    else:
        span_index = indexs[last_index:next_index]

    columns = data_set.values[span_index]
    real_C = columns[:, :3100]
    imag_C = columns[:, 3100: 6200]
    others = columns[:, 6200: 6241]
    label_pos = span_num - 21
    labels = columns[:, label_pos]
    return (next_index, {'real_C': real_C, 'imag_C': imag_C, 'others': others, 'labels': labels}, out_of_dataset)

###########################################################
################# graph helper ############################
###########################################################
#for create convolution kernel
def conv2d(x, W, stride, padding):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

#for create the pooling
def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#create weights
#important
#the weight initial value stddev is kinds of hyper para
#if a wrong stddev will stuck the network
def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.35)
  return tf.Variable(initial)


#create biases
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#to do the evaluation part for the whole data
#not use all data together, but many batchs
def do_eval(sess, data_set, batch_size, correct_num, placeholders, merged, test_writer, if_summary, global_step):

    real_C_pl, imag_C_pl, others_pl, labels_pl = placeholders
    num_epoch = data_set.shape[0] / batch_size
    rest_data_size = data_set.shape[0] % batch_size

    index = 0
    count = 0.0
    indexs = np.arange(data_set.shape[0])

    for step in xrange(num_epoch):
        index, data, _ = sequence_get_data(data_set, indexs, index, batch_size)

        if step == num_epoch - 1:
            if if_summary:
                summary, num = sess.run([merged, correct_num], feed_dict={real_C_pl: data['real_C'], imag_C_pl: data['imag_C'], others_pl: data['others'], labels_pl: data['labels'], keep_prob: 1.0})
                #add summary
                test_writer.add_summary(summary, global_step)
            else:
                num = sess.run(correct_num, feed_dict={real_C_pl: data['real_C'], imag_C_pl: data['imag_C'],others_pl: data['others'], labels_pl: data['labels'],keep_prob: 1.0})

        else:
            num = sess.run(correct_num, feed_dict={real_C_pl: data['real_C'], imag_C_pl: data['imag_C'],others_pl: data['others'], labels_pl: data['labels'],keep_prob: 1.0})

        count += num

    if rest_data_size != 0:
        #the rest data
        index, data, _ = sequence_get_data(data_set, indexs, index, rest_data_size)
        num = sess.run(correct_num, feed_dict={real_C_pl: data['real_C'], imag_C_pl: data['imag_C'],others_pl: data['others'], labels_pl: data['labels'], keep_prob: 1.0})

        count += num
    return count / data_set.shape[0]
