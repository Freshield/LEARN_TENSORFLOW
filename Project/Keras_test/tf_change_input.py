import pandas as pd
import tensorflow as tf
import numpy as np
import time
import os

################################################
#                                              #
#           use best model parameter           #
#      create model to train on the dataset    #
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

    train_set = dataset.values[0:-test_dataset_size * 2]
    validation_set = dataset.values[-test_dataset_size * 2:-test_dataset_size]
    test_set = dataset.values[-test_dataset_size:len(dataset)]


    return train_set, validation_set, test_set

#get a random data(maybe have same value)
def get_batch_data(X_dataset, y_dataset, batch_size):
    lines_num = X_dataset.shape[0] - 1
    random_index = np.random.randint(lines_num, size=[batch_size])

    X_data = X_dataset[random_index]
    y_data = y_dataset[random_index]
    return {'X': X_data, 'y': y_data}

#directly get whole dataset(only for small dataset)
def get_whole_data(X_dataset, y_dataset):

    return {'X': X_dataset, 'y': y_dataset}

#get a random indexs for dataset,
#so that we can shuffle the data every epoch
def get_random_seq_indexs(data_set):
    data_size = data_set.shape[0]
    #index = tf.random_shuffle(tf.range(0, data_size))#maybe can do it on tensorflow later
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    return indexs

#use the indexs together,
#so that we can sequence batch whole dataset
def sequence_get_data(X_dataset, y_dataset, indexs, last_index, batch_size):
    next_index = last_index + batch_size
    out_of_dataset = False

    if next_index > X_dataset.shape[0]:

        next_index -= X_dataset.shape[0]
        last_part = np.arange(last_index,indexs.shape[0])
        before_part = np.arange(next_index)
        span_index = indexs[np.concatenate((last_part, before_part))]
        out_of_dataset = True
    else:
        span_index = indexs[last_index:next_index]

    X_data = X_dataset[span_index]
    y_data = y_dataset[span_index]
    return (next_index, {'X':X_data,'y':y_data}, out_of_dataset)

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
  initial = tf.truncated_normal(shape, stddev=0.35,dtype=tf.float32)
  return tf.Variable(initial)


#create biases
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1,tf.float32, shape=shape)
  return tf.Variable(initial)

def batch_norm(bn_in):
    with tf.name_scope('BN'):
        offset = tf.Variable(tf.constant(0.0, shape=[bn_in.shape[-1]]), name='offset')
        scale = tf.Variable(tf.constant(1.0, shape=[bn_in.shape[-1]]), name='sclae')
        axises = np.arange(len(bn_in.shape) - 1)
        bn_mean, bn_var = tf.nn.moments(bn_in, axises, name='moment')
        normed = tf.nn.batch_normalization(bn_in, bn_mean, bn_var, offset, scale, 1e-3, name='bn_out')
    return normed

#to do the evaluation part for the whole data
#not use all data together, but many batchs
def do_eval(sess, X_dataset, y_dataset, batch_size, correct_num, placeholders, merged, test_writer, if_summary,
            global_step):

    input_x, input_y, keep_prob1, keep_prob2 = placeholders
    num_epoch = X_dataset.shape[0] / batch_size
    rest_data_size = X_dataset.shape[0] % batch_size

    index = 0
    count = 0.0
    indexs = np.arange(X_dataset.shape[0])

    for step in xrange(num_epoch):
        index, data, _ = sequence_get_data(X_dataset, y_dataset, indexs, index, batch_size)

        if step == num_epoch - 1:
            if if_summary:
                summary, num = sess.run([merged, correct_num], feed_dict={input_x:data['X'], input_y:data['y'],
                                                                          keep_prob1:1.0,keep_prob2:1.0})
                #add summary
                #test_writer.add_summary(summary, global_step)
            else:
                num = sess.run(correct_num, feed_dict={input_x:data['X'], input_y:data['y'],keep_prob1:1.0,keep_prob2:1.0})

        else:
            num = sess.run(correct_num, feed_dict={input_x:data['X'], input_y:data['y'],keep_prob1:1.0,keep_prob2:1.0})

        count += num

    if rest_data_size != 0:
        #the rest data
        index, data, _ = sequence_get_data(X_dataset, y_dataset, indexs, index, rest_data_size)
        num = sess.run(correct_num, feed_dict={input_x:data['X'], input_y:data['y'],keep_prob1:1.0,keep_prob2:1.0})

        count += num
    return count / X_dataset.shape[0]


def reshape_and_norm_dataset(dataset, SPAN):
    input_data = np.zeros((dataset.shape[0], 31, 100, 3))
    temp_data = np.reshape(dataset[:, :6200], (dataset.shape[0], 31, 100, 2))
    input_data[:, :, :, 0] = temp_data[:, :, :, 0]
    input_data[:, :, :, 1] = temp_data[:, :, :, 1]
    input_data[:, :, :, 2] = np.reshape(np.tile(dataset[:, 6200:6241], 76)[:, :3100], (dataset.shape[0], 31, 100))

    output_data = dataset[:, 6240 + SPAN[0]]

    # normalize data
    input_data[:, :, :, :2] = input_data[:, :, :, :2] - np.amin(input_data[:, :, :, :2])
    input_data[:, :, :, :2] = input_data[:, :, :, :2] / np.amax(input_data[:, :, :, :2])
    input_data[:, :, :, 2] = input_data[:, :, :, 2] - np.amin(input_data[:, :, :, 2])
    input_data[:, :, :, 2] = input_data[:, :, :, 2] / np.amax(input_data[:, :, :, 2])

    return input_data, output_data


############################################################
############### test #######################################
############################################################

#copy from cnn
NROWS = 10000 # for smaller datasets, choose from 100, 1000, 10000, and 'all'

SPAN=[20]

filename = '~/Ciena_data/ciena10000.csv'

data = pd.read_csv(filename, header=None, nrows=NROWS)

print "Data Shape: %s" % str(data.shape)


# Split into training and testing
train_set, validation_set, test_set = split_dataset(data, radio=0.1)

X_train, y_train = reshape_and_norm_dataset(train_set, SPAN)
X_test, y_test = reshape_and_norm_dataset(test_set, SPAN)

print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape

# max and min values of X training data

print np.amax(X_train)
print np.amin(X_train)
print X_train.dtype
print y_train.dtype


num_classes = 3
lr_rate = 0.01
reg = 0
max_step = 10000
batch_size = 64

print '-------------------now changed-----------------'
print 'lr_rate is', lr_rate
print 'reg is', reg
print '------------------------------------------------'

situation_now = '\n-------------------now changed-----------------\n' \
                'lr_rate is %.3f\nreg is %.3f\n' \
                '------------------------------------------------\n' % (lr_rate, reg)

with tf.Graph().as_default():
    with tf.Session() as sess:
        # inputs
        input_x = tf.placeholder(tf.float32, [None, 31, 100, 3])
        input_y = tf.placeholder(tf.int32, [None])
        input_y_one_hot = tf.one_hot(input_y, 3)
        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)

        #add image to summary so that you can see it in tensorboard
        #tf.summary.image('input', input_x, 20)

        bn_input = batch_norm(input_x)

        # build graph
        #convolution layer 1
        with tf.name_scope('conv1'):

            W_conv1 = tf.get_variable('weights1', [3, 3, 3, 32],initializer=tf.contrib.layers.xavier_initializer())
            #constraints
            W_conv1 = tf.minimum(W_conv1, 3)
            b_conv1 = bias_variable([32])
            h_conv1 = conv2d(input_x, W_conv1, 1, 'SAME') + b_conv1
            bn_conv1 = batch_norm(h_conv1)
            act_conv1 = tf.nn.relu(h_conv1)
            h_conv1_drop = tf.nn.dropout(act_conv1, keep_prob1)


        #convolution layer2
        with tf.name_scope('conv2'):
            W_conv2 = tf.get_variable('weights2', [3, 3, 32, 32],initializer=tf.contrib.layers.xavier_initializer())
            #constraints
            W_conv2 = tf.minimum(W_conv2, 3)
            b_conv2 = bias_variable([32])
            h_conv2 = conv2d(h_conv1_drop, W_conv2, 1, 'SAME') + b_conv2
            bn_conv2 = batch_norm(h_conv2)
            act_conv2 = tf.nn.relu(h_conv2)

            #16,50,32
            h_pool = max_pool_2x2(act_conv2)

        #fully connect layer1
        with tf.name_scope('fc1'):
            W_fc1 = tf.get_variable('weights3', [16 * 50 * 32, 512],initializer=tf.contrib.layers.xavier_initializer())
            # constraints
            W_fc1 = tf.minimum(W_fc1, 3)
            b_fc1 = bias_variable([512])
            #flatten the matrix
            h_pool_flat = tf.reshape(h_pool, [-1, 16 * 50 * 32])
            h_fc1 = tf.matmul(h_pool_flat, W_fc1) + b_fc1
            bn_fc1 = batch_norm(h_fc1)
            act_fc1 = tf.nn.relu(h_fc1)
            h_fc1_drop = tf.nn.dropout(act_fc1, keep_prob2)

        #the scores
        with tf.name_scope('scores'):
            W_fc2 = tf.get_variable('weights4', [512, 3],initializer=tf.contrib.layers.xavier_initializer())
            b_fc2 = bias_variable([3])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        #softmax
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y_one_hot, logits=y_conv),
            name='xentropy')

        # loss
        loss = cross_entropy

        train_step = tf.train.MomentumOptimizer(lr_rate, 0.9).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(input_y_one_hot, 1))
        correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        merged = tf.summary.merge_all()

        placeholders = (input_x, input_y, keep_prob1, keep_prob2)

        #writer = tf.summary.FileWriter('graph/', sess.graph)

        sess.run(tf.global_variables_initializer())


        test_path = 'tmp/test/logs'
        if tf.gfile.Exists(test_path):
            tf.gfile.DeleteRecursively(test_path)
        tf.gfile.MakeDirs(test_path)
        test_writer = tf.summary.FileWriter(test_path, sess.graph)

        # train
        # Train


        indexs = get_random_seq_indexs(X_train)
        out_of_dataset = False
        last_index = 0
        #saver = tf.train.Saver()

        for step in range(max_step):
            before_time = time.time()

            if out_of_dataset == True:
                indexs = get_random_seq_indexs(X_train)
                last_index = 0
                out_of_dataset = False

            last_index, data, out_of_dataset = sequence_get_data(X_train, y_train, indexs, last_index, batch_size)

            _, loss_v = sess.run([train_step, loss], feed_dict={input_x:data['X'], input_y:data['y'], keep_prob1:0.8,
                                                                keep_prob2:0.5})

            if step % 100 == 0:
                print 'loss in step %d is %f' % (step, loss_v)


            if step % 500 == 0 or step == max_step - 1:
                last_time = time.time()
                span_time = last_time - before_time
                print ('last 500 loop use %f sec' % (span_time * 500))
                print ('rest time is %f minutes' % (span_time * (max_step - step)/ 60))

                result = do_eval(sess, X_train, y_train, batch_size, correct_num, placeholders, merged, test_writer, False,
                                 step)
                print '----------train acc in step %d is %f-------------' % (step, result)
                result = do_eval(sess, X_test, y_test, batch_size, correct_num, placeholders, merged, test_writer,
                                 False, step)
                print '----------accuracy in step %d is %f-------------' % (step, result)

        result = do_eval(sess, X_test, y_test, batch_size, correct_num, placeholders, merged, test_writer, False, step)
        print '-----------last accuracy is %f------------' % (result)
