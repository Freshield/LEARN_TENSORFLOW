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

def get_random_seq_indexs(data_set):
    data_size = data_set.shape[0]
    #index = tf.random_shuffle(tf.range(0, data_size))
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    return indexs

def sequence_get_data(data_set, indexs, last_index, batch_size):
    next_index = last_index + batch_size
    out_of_dataset = False

    if next_index > data_set.shape[0]:
        last_index -= data_set.shape[0]
        next_index -= data_set.shape[0]
        out_of_dataset = True

    span_index = indexs[last_index:next_index]

    columns = data_set.values[span_index]
    real_C = columns[:, :3100]
    imag_C = columns[:, 3100: 6200]
    others = columns[:, 6200: 6241]
    labels = columns[:, -1]
    return (next_index, {'real_C': real_C, 'imag_C': imag_C, 'others': others, 'labels': labels}, out_of_dataset)

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

def do_eval(sess, data_set, batch_size, correct_num, placeholders, merged, test_writer, if_summary, global_step):
    real_C_pl, imag_C_pl, others_pl, labels_pl = placeholders
    num_epoch = data_set.shape[0] / batch_size
    reset_data_size = data_set.shape[0] % batch_size

    index = 0
    count = 0.0
    indexs = np.arange(data_set.shape[0])

    for step in xrange(num_epoch):
        index, data, _ = sequence_get_data(data_set, indexs, index, batch_size)

        if step == num_epoch - 1:
            if if_summary:
                summary, num = sess.run([merged, correct_num], feed_dict={real_C_pl: data['real_C'], imag_C_pl: data[
                'imag_C'], others_pl: data['others'], labels_pl: data['labels'], keep_prob: 1.0})
                test_writer.add_summary(summary, global_step)
            else:
                num = sess.run(correct_num, feed_dict={real_C_pl: data['real_C'], imag_C_pl: data['imag_C'],
                                                       others_pl: data['others'], labels_pl: data['labels'],
                                                       keep_prob: 1.0})
        else:
            num = sess.run(correct_num, feed_dict={real_C_pl: data['real_C'], imag_C_pl: data['imag_C'],
                                                   others_pl: data['others'], labels_pl: data['labels'],
                                                   keep_prob: 1.0})

        count += num
    if reset_data_size != 0:
        #the reset data
        index, data, _ = sequence_get_data(data_set, indexs, index, reset_data_size)
        num = sess.run(correct_num, feed_dict={real_C_pl: data['real_C'], imag_C_pl: data['imag_C'],
                                               others_pl: data['others'], labels_pl: data['labels'], keep_prob: 1.0})

        count += num
    return count / data_set.shape[0]


############################################################
############### test #######################################
############################################################

filename = 'ciena10000.csv'

dataset = pd.read_csv(filename, header=None)

train_dataset, validation_dataset, test_dataset = split_dataset(dataset, radio=0.1)

data = get_batch_data(train_dataset, 100)

batch_size = 100
lr_rate = 1e-3
max_step = 2000
keep_prob_v = 0.5
conv1_depth = 64
conv2_depth = 128
conv3_depth = 256
fc1_size = 2048
fc2_size = 512
reg = 0.02

loop_num = 1

print '-------------------now changed-----------------'
print 'conv1 is', conv1_depth
print 'conv2 is', conv2_depth
print 'conv3 is', conv3_depth
print 'keep_prob', keep_prob_v
print '------------------------------------------------'

situation_now = '\n-------------------now changed-----------------\n' \
                'conv1 is %d\nconv2 is %d\nconv3 is %d\nkeep_prob is %.2f\n' \
                '------------------------------------------------' % (
                    conv1_depth, conv2_depth, conv3_depth, keep_prob_v)

with tf.Graph().as_default():
    with tf.Session() as sess:
        # inputs
        real_C_pl = tf.placeholder(tf.float32, [None, 3100])
        imag_C_pl = tf.placeholder(tf.float32, [None, 3100])
        labels_pl = tf.placeholder(tf.int32, [None])
        keep_prob = tf.placeholder(tf.float32)
        others_pl = tf.placeholder(tf.float32, [None, 41])

        # reshape
        real_C_reshape = tf.reshape(real_C_pl, [-1, 31, 100, 1])
        imag_C_reshape = tf.reshape(imag_C_pl, [-1, 31, 100, 1])
        image_no_padding = tf.concat([real_C_reshape, imag_C_reshape], axis=1)
        others_r = tf.reshape(others_pl, shape=[-1, 1, 41, 1])
        others_r = tf.concat([others_r, others_r], axis=2)
        others_r_p = tf.pad(others_r, [[0,0],[0,0],[0,18],[0,0]], 'CONSTANT')
        others_reshape = tf.concat([others_r_p, others_r_p, others_r_p], axis=1)

        tf.summary.image('input', image_no_padding, 20)

        # tensors
        images = tf.concat([others_reshape, image_no_padding, others_reshape], axis=1)
        labels_one_hot = tf.one_hot(labels_pl, 3)

        # build graph

        W_conv1 = weight_variable([3, 3, 1, conv1_depth])
        b_conv1 = bias_variable([conv1_depth])
        h_conv1 = tf.nn.relu(conv2d(images, W_conv1, 1, 'SAME') + b_conv1)

        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([3, 3, conv1_depth, conv2_depth])
        b_conv2 = bias_variable([conv2_depth])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 1, 'SAME') + b_conv2)

        h_pool2 = max_pool_2x2(h_conv2)

        h_pool2_pad = tf.pad(h_pool2, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')

        W_conv3 = weight_variable([1, 1, conv2_depth, conv3_depth])
        b_conv3 = bias_variable([conv3_depth])
        h_conv3 = tf.nn.relu(conv2d(h_pool2_pad, W_conv3, 2, 'VALID') + b_conv3)

        h_pool3 = max_pool_2x2(h_conv3)

        W_fc1 = weight_variable([5 * 7 * conv3_depth, fc1_size])
        b_fc1 = bias_variable([fc1_size])

        h_pool3_flat = tf.reshape(h_pool3, [-1, 5 * 7 * conv3_depth])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

        W_fc2 = weight_variable([fc1_size, fc2_size])
        b_fc2 = bias_variable([fc2_size])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        W_fc3 = weight_variable([fc2_size, 3])
        b_fc3 = bias_variable([3])

        y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=y_conv),
            name='xentropy')

        reg_loss = 0.5 * reg * (
            tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
            tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) +
            tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) +
            tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
            tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2) +
            tf.nn.l2_loss(W_fc3) + tf.nn.l2_loss(b_fc3)
        )

        #loss = cross_entropy
        loss = cross_entropy + reg_loss

        train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels_one_hot, 1))
        correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        merged = tf.summary.merge_all()

        placeholders = (real_C_pl, imag_C_pl, others_pl, labels_pl)

        sess.run(tf.global_variables_initializer())

        test_path = 'tmp/test/logs'
        if tf.gfile.Exists(test_path):
            tf.gfile.DeleteRecursively(test_path)
        tf.gfile.MakeDirs(test_path)
        test_writer = tf.summary.FileWriter(test_path, sess.graph)

        # train
        # Train
        best_accuracy = 0.0

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

            _, loss_v = sess.run([train_step, loss],
                                 feed_dict={real_C_pl: data['real_C'], imag_C_pl: data['imag_C'],
                                            others_pl: data['others'], labels_pl: data['labels'],
                                            keep_prob: keep_prob_v})

            if step % 100 == 0:
                print 'loss in step %d is %f' % (step, loss_v)
                last_time = time.time()
                span_time = last_time - before_time
                print ('last 100 loop use %f sec' % (span_time * 100))
                print ('rest time is %f minutes' % (span_time * (max_step - step) * loop_num / 60))

            if step % 500 == 0 or step == max_step - 1:
                result = do_eval(sess, train_dataset, batch_size, correct_num, placeholders, merged, test_writer, False,
                                 step)
                print '----------train acc in step %d is %f-------------' % (step, result)
                result = do_eval(sess, validation_dataset, batch_size, correct_num, placeholders, merged, test_writer,
                                 True, step)
                print '----------accuracy in step %d is %f-------------' % (step, result)
                if result > best_accuracy:
                    best_accuracy = result

        result = do_eval(sess, test_dataset, batch_size, correct_num, placeholders, merged, test_writer, False, step)
        print '-----------last accuracy is %f------------' % (result)

        filename = '%.2f-%s' % (best_accuracy, situation_now)
        f = file(filename, 'w+')
        f.write(str(best_accuracy))
        f.write(situation_now)
        f.close()

        test_writer.close()

        loop_num -= 1
