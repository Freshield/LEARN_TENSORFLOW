import pandas as pd
import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
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
def get_batch_data(data_set, batch_size):
    lines_num = data_set.shape[0]
    random_index = np.random.randint(lines_num, size=[batch_size])
    columns = data_set[random_index]
    real_C = columns[:, :3100]
    imag_C = columns[:, 3100 : 6200]
    others = columns[:, 6200 : 6241]
    labels = columns[:, -2]

    return {'real_C':real_C, 'imag_C':imag_C, 'others':others, 'labels':labels}

#directly get whole dataset(only for small dataset)
def get_whole_data(data_set):
    real_C = data_set[:, :3100]
    imag_C = data_set[:, 3100: 6200]
    others = data_set[:, 6200: 6241]
    labels = data_set[:, -2]

    return {'real_C': real_C, 'imag_C': imag_C, 'others': others, 'labels': labels}

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
def sequence_get_data(data_set, indexs, last_index, batch_size):
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

    columns = data_set[span_index]
    real_C = columns[:, :3100]
    imag_C = columns[:, 3100: 6200]
    others = columns[:, 6200: 6241]
    labels = columns[:, -2]
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
def weight_variable(shape, name):
  """weight_variable generates a weight variable of a given shape."""
  weight = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

  return weight


#create biases
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def batch_norm_layer3(x, train_phase, scope_bn):
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = np.arange(len(x.shape) - 1)
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def batch_norm_layer2(x,train_phase,scope_bn):
    return batch_norm(x, decay=0.99, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)

def batch_norm_layer1(x,train_phase,scope_bn):
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    is_training=True,
    reuse=None,
    trainable=True,
    scope=scope_bn)

    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    is_training=False,
    reuse=True,
    trainable=True,
    scope=scope_bn)

    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z

#to do the evaluation part for the whole data
#not use all data together, but many batchs
def do_eval(sess, data_set, batch_size, correct_num, placeholders, merged=None, writer=None, global_step=None):

    real_C_pl, imag_C_pl, others_pl, labels_pl, keep_prob, train_phase = placeholders
    num_epoch = data_set.shape[0] / batch_size
    rest_data_size = data_set.shape[0] % batch_size

    index = 0
    count = 0.0
    indexs = np.arange(data_set.shape[0])

    for step in xrange(num_epoch):
        index, data, _ = sequence_get_data(data_set, indexs, index, batch_size)

        feed_dict = {real_C_pl: data['real_C'], imag_C_pl: data['imag_C'], others_pl: data['others'],
        labels_pl: data['labels'], keep_prob: keep_prob_v, train_phase: False}

        if merged != None:
            summary, num = sess.run([merged, correct_num], feed_dict=feed_dict)
            test_writer.add_summary(summary, global_step)
        else:
            num = sess.run(correct_num, feed_dict=feed_dict)

        count += num

    if rest_data_size != 0:
        #the rest data
        index, data, _ = sequence_get_data(data_set, indexs, index, rest_data_size)

        feed_dict = {real_C_pl: data['real_C'], imag_C_pl: data['imag_C'], others_pl: data['others'],
        labels_pl: data['labels'], keep_prob: 1.0, train_phase: False}

        if merged != None:
            summary, num = sess.run([merged, correct_num], feed_dict=feed_dict)
            test_writer.add_summary(summary, global_step)
        else:
            num = sess.run(correct_num, feed_dict=feed_dict)

        count += num
    return count / data_set.shape[0]


def normalize_dataset(dataset, mean_value=None):
    norm_dataset = np.zeros((dataset.shape))
    norm_dataset[:, :] = dataset[:, :]

    if mean_value == None:
        real_mean_value = np.mean(norm_dataset[:,:3100])
        imag_mean_value = np.mean(norm_dataset[:,3100:6200])
    else:
        real_mean_value, imag_mean_value = mean_value

    norm_dataset[:,:3100] -= real_mean_value
    norm_dataset[:,3100:6200] -= imag_mean_value

    return norm_dataset, (real_mean_value, imag_mean_value)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

# ensure the path exist
def del_and_create_dir(dir_path):
    if tf.gfile.Exists(dir_path):
        tf.gfile.DeleteRecursively(dir_path)
    tf.gfile.MakeDirs(dir_path)


############################################################
############### test #######################################
############################################################

filename = '/home/freshield/Ciena_data/ciena1000.csv'

dataset = pd.read_csv(filename, header=None)

train_dataset, validation_dataset, test_dataset = split_dataset(dataset, radio=0.1)

train_dataset, train_mean = normalize_dataset(train_dataset)

validation_dataset, _ = normalize_dataset(validation_dataset, train_mean)

test_dataset, _ = normalize_dataset(test_dataset, train_mean)

#data = get_batch_data(train_dataset, 100)

batch_size = 100
lr_rate = 0.015
max_step =2000
keep_prob_v = 1.0
conv1_depth = 64
conv2_depth = 128
conv3_depth = 256
fc1_size = 512
fc2_size = 256
reg = 1e-4
lr_decay = 0.99
lr_loop = 1000

loop_num = 1

print '-------------------now changed-----------------'
print 'lr_rate is', lr_rate
print 'reg is', reg
print 'keep_prob', keep_prob_v
print '------------------------------------------------'

situation_now = '\n-------------------now changed-----------------\n' \
                'lr_rate is %.3f\nreg is %.3f\nkeep_prob is %.2f\n' \
                '------------------------------------------------\n' % (lr_rate, reg, keep_prob_v)

with tf.Graph().as_default():
    with tf.Session() as sess:
        # inputs
        real_C_pl = tf.placeholder(tf.float32, [None, 3100])
        imag_C_pl = tf.placeholder(tf.float32, [None, 3100])
        labels_pl = tf.placeholder(tf.int32, [None])
        keep_prob = tf.placeholder(tf.float32)
        others_pl = tf.placeholder(tf.float32, [None, 41])
        train_phase = tf.placeholder(tf.bool, name='train_phase')

        # reshape
        real_C_reshape = tf.reshape(real_C_pl, [-1, 31, 100, 1])
        imag_C_reshape = tf.reshape(imag_C_pl, [-1, 31, 100, 1])
        image_no_padding = tf.concat([real_C_reshape, imag_C_reshape], axis=1)
        others_r = tf.reshape(others_pl, shape=[-1, 1, 41, 1])
        others_r = tf.concat([others_r, others_r], axis=2)
        others_r_p = tf.pad(others_r, [[0, 0], [0, 0], [0, 18], [0, 0]], 'CONSTANT')
        others_reshape = tf.concat([others_r_p, others_r_p, others_r_p], axis=1)

        #put the others and image together
        images = tf.concat([others_reshape, image_no_padding, others_reshape], axis=1)
        labels_one_hot = tf.one_hot(labels_pl, 3)

        bn_input = batch_norm_layer3(images, train_phase, 'bn_input')

        # build graph
        #convolution layer 1
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([3, 3, 1, conv1_depth], 'w1')
            W_conv1 = tf.minimum(W_conv1, 3)
            variable_summaries(W_conv1)
            b_conv1 = bias_variable([conv1_depth])
            h_conv1 = conv2d(bn_input, W_conv1, 1, 'SAME') + b_conv1
            bn_conv1 = batch_norm_layer3(h_conv1, train_phase, 'bn_conv1')
            act_conv1 = tf.nn.relu(bn_conv1)
            tf.summary.histogram('act_conv1',act_conv1)

            h_pool1 = max_pool_2x2(act_conv1)
        #convolution layer2
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([3, 3, conv1_depth, conv2_depth], 'w2')
            W_conv2 = tf.minimum(W_conv2, 3)
            variable_summaries(W_conv2)
            b_conv2 = bias_variable([conv2_depth])
            h_conv2 = conv2d(h_pool1, W_conv2, 1, 'SAME') + b_conv2
            bn_conv2 = batch_norm_layer3(h_conv2, train_phase, 'bn_conv2')
            act_conv2 = tf.nn.relu(bn_conv2)
            tf.summary.histogram('act_conv2',act_conv2)

            h_pool2 = max_pool_2x2(act_conv2)
        #convolution layer3
        with tf.name_scope('conv3'):
            h_pool2_pad = tf.pad(h_pool2, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')

            W_conv3 = weight_variable([1, 1, conv2_depth, conv3_depth], 'w3')
            W_conv3 = tf.minimum(W_conv3, 3)
            variable_summaries(W_conv3)
            b_conv3 = bias_variable([conv3_depth])
            h_conv3 = conv2d(h_pool2_pad, W_conv3, 2, 'VALID') + b_conv3
            bn_conv3 = batch_norm_layer3(h_conv3, train_phase, 'bn_conv3')
            act_conv3 = tf.nn.relu(bn_conv3)
            tf.summary.histogram('act_conv3',act_conv3)

            h_pool3 = max_pool_2x2(act_conv3)
        #fully connect layer1
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([5 * 7 * conv3_depth, fc1_size], 'w4')
            W_fc1 = tf.minimum(W_fc1, 3)
            variable_summaries(W_fc1)
            b_fc1 = bias_variable([fc1_size])
            #flatten the matrix
            h_pool3_flat = tf.reshape(h_pool3, [-1, 5 * 7 * conv3_depth])
            h_fc1 = tf.matmul(h_pool3_flat, W_fc1) + b_fc1
            bn_fc1 = batch_norm_layer3(h_fc1, train_phase, 'bn_fc1')
            act_fc1 = tf.nn.relu(bn_fc1)
            tf.summary.histogram('act_fc1',act_fc1)
        #fully connect layer2
        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([fc1_size, fc2_size], 'w5')
            W_fc2 = tf.minimum(W_fc2, 3)
            variable_summaries(W_fc2)
            b_fc2 = bias_variable([fc2_size])
            h_fc2 = tf.matmul(act_fc1, W_fc2) + b_fc2
            bn_fc2 = batch_norm_layer3(h_fc2, train_phase, 'bn_fc2')
            act_fc2 = tf.nn.relu(bn_fc2)
            tf.summary.histogram('act_fc2',act_fc2)

            h_fc2_drop = tf.nn.dropout(act_fc2, keep_prob=keep_prob)
        #the scores
        with tf.name_scope('fc3'):
            W_fc3 = weight_variable([fc2_size, 3], 'w6')
            W_fc3 = tf.minimum(W_fc3, 3)
            variable_summaries(W_fc3)
            b_fc3 = bias_variable([3])

            y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
        #softmax
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=y_conv),
            name='xentropy')
        #L2 regularzation
        reg_loss = 0.5 * reg * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2) + tf.nn.l2_loss(W_fc3) + tf.nn.l2_loss(b_fc3))

        # loss
        loss = cross_entropy + reg_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels_one_hot, 1))
        correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('acc', accuracy)

        placeholders = (real_C_pl, imag_C_pl, others_pl, labels_pl, keep_prob, train_phase)

        merged = tf.summary.merge_all()

        # define store path
        train_path = 'tmp/train'
        test_path = 'tmp/test'
        del_and_create_dir(train_path)
        del_and_create_dir(test_path)

        # create writer for tensorboard
        train_writer = tf.summary.FileWriter(train_path, sess.graph)
        test_writer = tf.summary.FileWriter(test_path)

        sess.run(tf.global_variables_initializer())

        indexs = get_random_seq_indexs(train_dataset)
        out_of_dataset = False
        last_index = 0

        for step in range(max_step):
            before_time = time.time()

            if out_of_dataset == True:
                indexs = get_random_seq_indexs(train_dataset)
                last_index = 0
                out_of_dataset = False

            last_index, data, out_of_dataset = sequence_get_data(train_dataset, indexs, last_index, batch_size)

            feed_dict = {real_C_pl: data['real_C'], imag_C_pl: data['imag_C'],
            others_pl: data['others'], labels_pl: data['labels'],
            keep_prob: keep_prob_v, train_phase:True}


            if step % 40 == 0 or step == max_step -1:
                summary, _, loss_v, acc = sess.run([merged, train_step, loss, accuracy], feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
            else:
                _, loss_v, acc = sess.run([train_step, loss, accuracy], feed_dict=feed_dict)

            if step % 100 == 0:
                print 'loss in step %d is %f, acc is %.3f' % (step, loss_v, acc)
            else:
                print 'acc in step %d is %.3f' % (step, acc)

            if step % 300 == 0 or step == max_step - 1:
                last_time = time.time()
                span_time = last_time - before_time
                print ('last 100 loop use %f sec' % (span_time * 100))
                print ('rest time is %f minutes' % (span_time * (max_step - step) * loop_num / 60))

                result = do_eval(sess, train_dataset, batch_size, correct_num, placeholders)
                #acc_data = get_whole_data(train_dataset)

                #feed_dict = {real_C_pl: acc_data['real_C'], imag_C_pl: acc_data['imag_C'], others_pl: acc_data[
                # 'others'],labels_pl: acc_data['labels'], keep_prob: 1.0, train_phase: True}

                #result = sess.run(accuracy, feed_dict=feed_dict)

                print '----------train acc in step %d is %f-------------' % (step, result)
                result = do_eval(sess, validation_dataset, batch_size, correct_num, placeholders, merged,test_writer, step)
                #acc_data = get_whole_data(validation_dataset)

                #feed_dict = {real_C_pl: acc_data['real_C'], imag_C_pl: acc_data['imag_C'],
                #             others_pl: acc_data['others'], labels_pl: acc_data['labels'], keep_prob: 1.0,
                #             train_phase: True}

                #summary, result = sess.run([merged, accuracy], feed_dict=feed_dict)
                #test_writer.add_summary(summary, step)

                print '----------accuracy in step %d is %f-------------' % (step, result)

            if step > 0 and step % lr_loop == 0:
                lr_rate *= lr_decay

        result = do_eval(sess, test_dataset, batch_size, correct_num, placeholders)
        print '-----------last accuracy is %f------------' % (result)
