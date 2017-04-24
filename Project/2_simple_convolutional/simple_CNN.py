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

    return (next_index,{'real_C': real_C, 'imag_C': imag_C, 'others': others, 'labels': labels})

###########################################################
################# graph helper ############################
###########################################################
def conv2d(x, W, stride, padding):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def do_eval(sess, data_set, batch_size, accuracy, placeholders, merged, test_writer, if_summary, global_step):
    real_C_pl, imag_C_pl, others_pl, labels_pl = placeholders
    num_epoch = len(data_set) / batch_size
    reset_data_size = len(data_set) % batch_size

    index = 0
    count = 0.0
    for step in xrange(num_epoch):
        index, data = sequence_get_data(data_set, index, batch_size)

        if step == num_epoch - 1:
            if if_summary:
                summary, result = sess.run([merged, accuracy], feed_dict={real_C_pl: data['real_C'], imag_C_pl: data[
                'imag_C'], others_pl: data['others'], labels_pl: data['labels'], keep_prob: 1.0})
                test_writer.add_summary(summary, global_step)
            else:
                result = sess.run(accuracy, feed_dict={real_C_pl: data['real_C'], imag_C_pl: data['imag_C'],
                                                       others_pl: data['others'], labels_pl: data['labels'],
                                                       keep_prob: 1.0})
        else:
            result = sess.run(accuracy, feed_dict={real_C_pl: data['real_C'], imag_C_pl: data['imag_C'],
                                                   others_pl: data['others'], labels_pl: data['labels'],
                                                   keep_prob: 1.0})

        count += result * batch_size
    if reset_data_size != 0:
        #the reset data
        index, data = sequence_get_data(data_set, index, reset_data_size)
        result = sess.run(accuracy, feed_dict={real_C_pl: data['real_C'], imag_C_pl: data['imag_C'],
                                               others_pl: data['others'], labels_pl: data['labels'], keep_prob: 1.0})

        count += result * reset_data_size
    return count / len(data_set)


############################################################
############### test #######################################
############################################################

filename = 'ciena1000.csv'

dataset = pd.read_csv(filename, header=None)

print dataset.values.shape

train_dataset, validation_dataset, test_dataset = split_dataset(dataset, radio=0.1)

data = get_batch_data(train_dataset, 100)

batch_size = 100
lr_rate = 1e-4
max_step = 20000
keep_prob_v = 0.5

with tf.Graph().as_default():
    with tf.Session() as sess:
        #inputs
        real_C_pl = tf.placeholder(tf.float32, [None, 3100])
        imag_C_pl = tf.placeholder(tf.float32, [None, 3100])
        labels_pl = tf.placeholder(tf.int32, [None])

        #reshape
        real_C_reshape = tf.reshape(real_C_pl, [-1, 31, 100, 1])
        imag_C_reshape = tf.reshape(imag_C_pl, [-1, 31, 100, 1])
        image_no_padding = tf.concat([real_C_reshape, imag_C_reshape], axis=1)

        tf.summary.image('input', image_no_padding, 20)

        #tensors
        images = tf.pad(image_no_padding, [[0,0], [3,3], [0,0], [0,0]], 'CONSTANT')
        labels_one_hot = tf.one_hot(labels_pl, 3)

        others_pl = tf.placeholder(tf.float32, [None, 41])

        #build graph

        W_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(images, W_conv1, 1, 'SAME') + b_conv1)

        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 1, 'SAME') + b_conv2)

        h_pool2 = max_pool_2x2(h_conv2)

        h_pool2_pad = tf.pad(h_pool2, [[0,0], [1,1], [1,1], [0,0]], 'CONSTANT')

        W_conv3 = weight_variable([1, 1, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_pool2_pad, W_conv3, 2, 'VALID') + b_conv3)

        h_pool3 = max_pool_2x2(h_conv3)


        W_fc1 = weight_variable([5 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool3_flat = tf.reshape(h_pool3, [-1, 5*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 3])
        b_fc2 = bias_variable([3])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=y_conv), name='xentropy')

        train_step = tf.train.AdamOptimizer(lr_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        merged = tf.summary.merge_all()

        placeholders = (real_C_pl, imag_C_pl, others_pl, labels_pl)

        sess.run(tf.global_variables_initializer())

        test_path = 'tmp/test/logs'
        if tf.gfile.Exists(test_path):
            tf.gfile.DeleteRecursively(test_path)
        tf.gfile.MakeDirs(test_path)
        test_writer = tf.summary.FileWriter(test_path, sess.graph)

        #train
        # Train
        for step in range(max_step):
            before_time = time.time()
            data = get_batch_data(train_dataset, batch_size)
            # data = get_whole_data(train_dataset)
            # index, data = sequence_get_data(train_dataset, index, batch)


            _, loss_v = sess.run([train_step, cross_entropy], feed_dict={real_C_pl:data['real_C'], imag_C_pl:data['imag_C'],
                          others_pl:data['others'], labels_pl:data['labels'], keep_prob:1.0})

            if step % 100 == 0:
                print 'loss in step %d is %f' % (step, loss_v)
                last_time = time.time()
                span_time = last_time - before_time
                print ('last 100 loop use %f sec' % (span_time * 100))
                print ('rest time is %f minutes' % (span_time * (max_step - step) / 60))

            if step % 500 == 0 or step == max_step - 1:
                result = do_eval(sess, train_dataset, batch_size, accuracy, placeholders,
                                 merged, test_writer, False, step)
                print '----------train acc in step %d is %f-------------' % (step, result)
                result = do_eval(sess, validation_dataset, batch_size, accuracy, placeholders,
                                 merged, test_writer, True, step)
                print '----------accuracy in step %d is %f-------------' % (step, result)

        result = do_eval(sess, test_dataset, batch_size, accuracy, placeholders,
                         merged, test_writer, False, step)
        print '-----------last accuracy is %f------------' % (result)

        test_writer.close()







