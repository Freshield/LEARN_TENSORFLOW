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
def split_dataset(dataset, radio):
    test_dataset_size = int(radio * len(dataset))
    data_size = dataset.shape[0]
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    train_set = dataset.values[indexs[:-test_dataset_size * 2]]
    validation_set = dataset.values[indexs[-test_dataset_size * 2 : -test_dataset_size]]
    test_set = dataset.values[indexs[-test_dataset_size : ]]
    return train_set, validation_set, test_set

def normalize_dataset(dataset, min_values=None, max_values=None):
    norm_dataset = np.zeros((dataset.shape))
    norm_dataset[:, :] = dataset[:, :]

    if min_values == None:
        CM_r_min = np.min(norm_dataset[:,:3100])
        CM_i_min = np.min(norm_dataset[:,3100:6200])
        CD_min = np.min(norm_dataset[:,6200:6201])
        length_min = np.min(norm_dataset[:,6201:6221])
        power_min = np.min(norm_dataset[:,6221:6241])
    else:
        CM_r_min, CM_i_min, CD_min, length_min, power_min = min_values


    if max_values == None:
        CM_r_max = np.max(norm_dataset[:,0:3100])
        CM_i_max = np.max(norm_dataset[:,3100:6200])
        CD_max = np.max(norm_dataset[:,6200:6201])
        length_max = np.max(norm_dataset[:,6201:6221])
        power_max = np.max(norm_dataset[:,6221:6241])
    else:
        CM_r_max, CM_i_max, CD_max, length_max, power_max = max_values

    def calcul_norm(dataset, min, max):
        return (dataset - min) / (max - min)


    norm_dataset[:,0:3100] = calcul_norm(norm_dataset[:,0:3100], CM_r_min, CM_r_max)
    norm_dataset[:,3100:6200] = calcul_norm(norm_dataset[:,3100:6200], CM_i_min, CM_i_max)
    norm_dataset[:,6200:6201] = calcul_norm(norm_dataset[:,6200:6201], CD_min, CD_max)
    norm_dataset[:,6201:6221] = calcul_norm(norm_dataset[:,6201:6221], length_min, length_max)
    norm_dataset[:,6221:6241] = calcul_norm(norm_dataset[:,6221:6241], power_min, power_max)

    min_values = (CM_r_min, CM_i_min, CD_min, length_min, power_min)
    max_values = (CM_r_max, CM_i_max, CD_max, length_max, power_max)

    return norm_dataset, min_values, max_values

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

#ensure the path exist
def del_and_create_dir(dir_path):
    if tf.gfile.Exists(dir_path):
        tf.gfile.DeleteRecursively(dir_path)
    tf.gfile.MakeDirs(dir_path)

#write log file
def write_file(result, dir_path, situation_now):
    filename = 'modules/%f-%s' % (result, dir_path)
    f = file(filename, 'w+')
    f.write(dir_path)
    f.write(situation_now)
    f.close()
    print 'best file writed'

###########################################################
################# graph helper ############################
###########################################################
#for summary the tensors
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

# create weights
# important
# the weight initial value stddev is kinds of hyper para
# if a wrong stddev will stuck the network
def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape=shape, stddev=stddev)

    return tf.Variable(initial, name='weights')

# create biases
def biases_variable(shape, value):
    initial = tf.constant(value=value, dtype=tf.float32, shape=shape)

    return tf.Variable(initial, name='biases')


#for create convolution kernel
def conv2d(x, W, stride, padding):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

#for create the pooling
def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#get input placeholders,
#also create one hot label here
def get_inputs(summary=True):
    with tf.name_scope('input'):
        # inputs
        real_C_pl = tf.placeholder(tf.float32, [None, 3100])
        imag_C_pl = tf.placeholder(tf.float32, [None, 3100])
        y_ = tf.placeholder(tf.int32, [None])
        others_pl = tf.placeholder(tf.float32, [None, 41])

        # reshape
        real_C_reshape = tf.reshape(real_C_pl, [-1, 31, 100, 1])
        imag_C_reshape = tf.reshape(imag_C_pl, [-1, 31, 100, 1])
        image_no_padding = tf.concat([real_C_reshape, imag_C_reshape], axis=1)
        others_r = tf.reshape(others_pl, shape=[-1, 1, 41, 1])
        others_r = tf.concat([others_r, others_r], axis=2)
        others_r_p = tf.pad(others_r, [[0, 0], [0, 0], [0, 18], [0, 0]], 'CONSTANT')
        others_reshape = tf.concat([others_r_p, others_r_p, others_r_p], axis=1)

        # put the others and image together
        images = tf.concat([others_reshape, image_no_padding, others_reshape], axis=1)
        y_one_hot = tf.one_hot(y_, 3)
        if summary == True:
            # add image to summary so that you can see it in tensorboard
            tf.summary.image('input', images, 20)
    return images, y_one_hot

#create convolution layer,
#return parameter for L2 regularzation
def get_conv_layer(input, input_depth, conv_depth, stddev, b_value, F, name='conv', S=1, padding='SAME',
                   act=tf.nn.relu, summary = True):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            W_conv = weight_variable([F, F, input_depth, conv_depth], stddev)
            if summary == True:
                variable_summaries(W_conv)
        with tf.name_scope('biases'):
            b_conv = biases_variable([conv_depth], b_value)
            if summary == True:
                variable_summaries(b_conv)
        with tf.name_scope('activation'):
            h_conv = act(conv2d(input, W_conv, S, padding) + b_conv, name='activation')
            if summary == True:
                tf.summary.histogram('activation', h_conv)
        with tf.name_scope('pooling'):
            h_pool = max_pool_2x2(h_conv)
            if summary == True:
                tf.summary.histogram('activation', h_conv)
        parameters = (W_conv, b_conv)

    return h_pool, parameters

#create fully connect layer,
# relu(x * W + b)
#return parameter for L2 regularzation
def get_fc_layer(input, input_size, hidden_size, stddev, b_value, name='hidden', act=tf.nn.relu, summary = True):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            W = weight_variable([input_size, hidden_size], stddev)
            if summary == True:
                variable_summaries(W)
        with tf.name_scope('biases'):
            b = biases_variable([hidden_size], b_value)
            if summary == True:
                variable_summaries(b)
        with tf.name_scope('activation'):
            activation = act(tf.matmul(input, W) + b, name='activation')
            if summary == True:
                tf.summary.histogram('activation', activation)

        parameters = (W, b)

    return activation, parameters

#important
#scores must different with hidden layer
#because there have relu at the end of hidden
#and score needn't
# x * W + b
#return parameter for L2 regularzation
def get_scores(input, input_size, hidden_size, stddev, b_value, name='scores', summary = True):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            W = weight_variable([input_size, hidden_size], stddev)
            if summary == True:
                variable_summaries(W)
        with tf.name_scope('biases'):
            b = biases_variable([hidden_size], b_value)
            if summary == True:
                variable_summaries(b)
        with tf.name_scope('scores'):
            y = tf.matmul(input, W) + b
            if summary == True:
                tf.summary.histogram('scores', y)

        parameters = (W, b)

    return y, parameters

#drop the input
def get_droppout(input):
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        dropout = tf.nn.dropout(input, keep_prob=keep_prob)
    return dropout, keep_prob

#3 layers neural network model
#input 241, output 3
def get_logits(x, conv1_depth, conv2_depth, conv3_depth, fc1_size, fc2_size, labels_size, stddev, b_value):

    conv1, conv1_para = get_conv_layer(x, 1, conv1_depth, stddev, stddev, 3, 'conv1')

    conv2, conv2_para = get_conv_layer(conv1, conv1_depth, conv2_depth, stddev, stddev, 3, 'conv2')

    conv2_pad = tf.pad(conv2, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')

    conv3, conv3_para = get_conv_layer(conv2_pad, conv2_depth, conv3_depth, stddev, stddev, 1, 'conv3', 2, 'VALID')

    conv3_flat = tf.reshape(conv3, [-1, 5 * 7 * conv3_depth])

    fc1, fc1_para = get_fc_layer(conv3_flat, 5 * 7 * conv3_depth, fc1_size, stddev, stddev, 'fc1')

    fc2, fc2_para = get_fc_layer(fc1, fc1_size, fc2_size, stddev, stddev, 'fc2')

    fc2_drop, keep_prob = get_droppout(fc2)

    y, scores_para = get_fc_layer(fc2_drop, fc2_size, labels_size, stddev, stddev, 'scores')

    total_parameters = [conv1_para, conv2_para, conv3_para, fc1_para, fc2_para, scores_para]

    return y, total_parameters, keep_prob


#get loss by softmax
#also can choose if use L2 regularzation or not
def get_loss(y, y_one_hot, total_parameters, reg, summary = True, use_l2_loss = True):
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=y), name='xentropy')
        if use_l2_loss == True:
            reg_loss = 0
            for parameter in total_parameters:
                W, b = parameter
                reg_loss += 0.5 * reg * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))
            loss = cross_entropy + reg_loss
        else:
            loss = cross_entropy

        if summary == True:
            tf.summary.scalar('cross_entropy', cross_entropy)
            tf.summary.scalar('loss', loss)

        return loss

# get train handle for training and backpropagation
# use adam function
def get_train_op(loss, lr_rate, optimizer=tf.train.AdamOptimizer):
    with tf.name_scope('train'):
        train_op = optimizer(lr_rate).minimize(loss)
    return train_op


#to do the evaluation part for the whole data
#not use all data together, but many batchs
def do_eval(sess, data_set, batch_size, correct_num, placeholders, merged, test_writer, if_summary, global_step):

    real_C_pl, imag_C_pl, others_pl, labels_pl, keep_prob = placeholders
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
