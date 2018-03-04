import tensorflow as tf
import numpy as np
import os
import pandas as pd
import time


############################################################
############# helpers ######################################
############################################################

# to copy files from source dir to target dir
def copyFiles(sourceDir, targetDir):
    if sourceDir.find(".csv") > 0:
        print 'error'
        return
    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir, file)
        targetFile = os.path.join(targetDir, file)
        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or (
                os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))):
                open(targetFile, "wb").write(open(sourceFile, "rb").read())
        if os.path.isdir(sourceFile):
            First_Directory = False
            copyFiles(sourceFile, targetFile)


# ensure the path exist
def del_and_create_dir(dir_path):
    if tf.gfile.Exists(dir_path):
        tf.gfile.DeleteRecursively(dir_path)
    tf.gfile.MakeDirs(dir_path)


# split the dataset into three part:
# training, validation, test
def split_dataset(dataset, test_dataset_size=None, radio=None):
    if radio != None:
        test_dataset_size = int(radio * len(dataset))

    train_set = dataset.values[0:-test_dataset_size * 2]
    validation_set = dataset.values[-test_dataset_size * 2:-test_dataset_size]
    test_set = dataset.values[-test_dataset_size:len(dataset)]

    return train_set, validation_set, test_set


# get a random data(maybe have same value)
def get_batch_data(X_dataset, para_dataset, y_dataset, batch_size):
    lines_num = X_dataset.shape[0]
    random_index = np.random.randint(lines_num, size=[batch_size])
    X_data = X_dataset[random_index]
    para_data = para_dataset[random_index]
    y_data = y_dataset[random_index]
    return {'X': X_data, 'p':para_data, 'y': y_data}


# directly get whole dataset(only for small dataset)
def get_whole_data(X_dataset, para_dataset, y_dataset):
    return {'X': X_dataset, 'p':para_dataset, 'y': y_dataset}


# get a random indexs for dataset,
# so that we can shuffle the data every epoch
def get_random_seq_indexs(data_set):
    data_size = data_set.shape[0]
    # index = tf.random_shuffle(tf.range(0, data_size))#maybe can do it on tensorflow later
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    return indexs


# get a random indexs for file,
# so that we can shuffle the data every epoch
def get_file_random_seq_indexs(num):
    indexs = np.arange(num)
    np.random.shuffle(indexs)
    return indexs


# use the indexs together,
# so that we can sequence batch whole dataset
def sequence_get_data(X_dataset, para_dataset, y_dataset, indexs, last_index, batch_size):
    next_index = last_index + batch_size
    out_of_dataset = False

    if next_index > X_dataset.shape[0]:

        next_index -= X_dataset.shape[0]
        last_part = np.arange(last_index, indexs.shape[0])
        before_part = np.arange(next_index)
        span_index = indexs[np.concatenate((last_part, before_part))]  # link two parts together
        out_of_dataset = True
    else:
        span_index = indexs[last_index:next_index]

    X_data = X_dataset[span_index]
    para_data = para_dataset[span_index]
    y_data = y_dataset[span_index]
    return (next_index, {'X': X_data, 'p':para_data, 'y': y_data}, out_of_dataset)


def normalize_dataset(dataset, min_values=None, max_values=None):
    norm_dataset = np.zeros((dataset.shape))
    norm_dataset[:, :] = dataset[:, :]

    if min_values == None:
        CMr_min = np.min(norm_dataset[:, 0:3100])
        CMi_min = np.min(norm_dataset[:, 3100:6200])
        CD_min = np.min(norm_dataset[:, 6200:6201])
        length_min = np.min(norm_dataset[:, 6201:6221])
        power_min = np.min(norm_dataset[:, 6221:6241])
    else:
        CMr_min, CMi_min, CD_min, length_min, power_min = min_values

    if max_values == None:
        CMr_max = np.max(norm_dataset[:, 0:3100])
        CMi_max = np.max(norm_dataset[:, 3100:6200])
        CD_max = np.max(norm_dataset[:, 6200:6201])
        length_max = np.max(norm_dataset[:, 6201:6221])
        power_max = np.max(norm_dataset[:, 6221:6241])
    else:
        CMr_max, CMi_max, CD_max, length_max, power_max = max_values

    def calcul_norm(dataset, min, max):
        return (dataset - min) / (max - min)

    norm_dataset[:, 0:3100] = calcul_norm(norm_dataset[:, 0:3100], CMr_min, CMr_max)
    norm_dataset[:, 3100:6200] = calcul_norm(norm_dataset[:, 3100:6200], CMi_min, CMi_max)
    norm_dataset[:, 6200:6201] = calcul_norm(norm_dataset[:, 6200:6201], CD_min, CD_max)
    norm_dataset[:, 6201:6221] = calcul_norm(norm_dataset[:, 6201:6221], length_min, length_max)
    norm_dataset[:, 6221:6241] = calcul_norm(norm_dataset[:, 6221:6241], power_min, power_max)

    min_values = (CMr_min, CMi_min, CD_min, length_min, power_min)
    max_values = (CMr_max, CMi_max, CD_max, length_max, power_max)

    return norm_dataset, min_values, max_values


def num_to_one_hot(dataset, category_num):
    lines = dataset.shape[0]
    one_hot_dataset = np.zeros([lines, category_num], dtype=np.float32)
    one_hot_dataset[np.arange(lines), dataset] = 1
    return one_hot_dataset


def reshape_dataset(dataset, SPAN):
    input_data = np.zeros((dataset.shape[0], 32, 104, 2))
    temp_data = np.reshape(dataset[:, :6200], (-1, 31, 100, 2))
    input_data[:, :31, 2:102, 0] = temp_data[:, :, :, 0]  # cause input size is 32 not 31
    input_data[:, :31, 2:102, 1] = temp_data[:, :, :, 1]
    para_data = dataset[:, 6200:6241]

    output_data = dataset[:, 6240 + SPAN[0]].astype(int)
    output_data = num_to_one_hot(output_data, 3)

    return input_data, para_data, output_data


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
    X_data, para_data, y_data = reshape_dataset(dataset.values, SPAN)
    return X_data, para_data, y_data


###########################################################
################# graph helper ############################
###########################################################

# create weights
def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    weight = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return weight


# create biases
def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


# for create convolution kernel
def conv2d(x, W, stride, padding):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


# for create the pooling
def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# for create batch norm layer
def batch_norm_layer(x, train_phase, scope_bn):
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


# conv-bn-relu-maxpooling
def conv_bn_pool_layer(input_layer, filter_depth, train_phase, name):
    input_depth = input_layer.shape[-1]
    with tf.variable_scope(name):
        filter = weight_variable([3, 3, input_depth, filter_depth], "filter")
        biases = bias_variable([filter_depth])
        conv_output = conv2d(input_layer, filter, 1, "SAME") + biases
        bn_output = batch_norm_layer(conv_output, train_phase, "conv_bn")
        act_output = tf.nn.relu(bn_output)
        output = max_pool_2x2(act_output)
    return output, filter


# fully connected layer
# fc-bn-relu-drop
def fc_bn_drop_layer(input_layer, output_size, train_phase, keep_prob, name):
    with tf.variable_scope(name):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, output_size], 'fc_weight')
        b = bias_variable([output_size])
        fc_out = tf.matmul(input_layer, W) + b
        bn_out = batch_norm_layer(fc_out, train_phase, "fc_bn")
        act_out = tf.nn.relu(bn_out)
        output = tf.nn.dropout(act_out, keep_prob)
    return output, W


# score layer
def score_layer(input_layer, label_size):
    with tf.variable_scope("score"):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, label_size], 'score_weight')
        b = bias_variable([label_size])
        output = tf.matmul(input_layer, W) + b
    return output, W


# get the y_pred
def inference(input_layer, para_data, train_phase, keep_prob):
    with tf.variable_scope("inference"):
        # input (N,32,104,2)
        bn_input = batch_norm_layer(input_layer, train_phase, "bn_input")

        # conv1 (N,16,52,64)
        conv1, filter1 = conv_bn_pool_layer(bn_input, 64, train_phase, "conv1")

        # conv2 (N,8,26,128)
        conv2, filter2 = conv_bn_pool_layer(conv1, 128, train_phase, "conv2")

        # conv3 (N, 4, 13, 256)
        conv3, filter3 = conv_bn_pool_layer(conv2, 256, train_phase, "conv3")

        # flat
        flat_conv3 = tf.reshape(conv3, [-1, 4 * 13 * 256])

        # fc layer1(N, 512)
        fc1, fc_weight1 = fc_bn_drop_layer(flat_conv3, 512, train_phase, keep_prob, "fc1")

        #link the para_data
        fc1_link = tf.concat([fc1, para_data], axis=1)

        #fc layer2(N,256)
        fc2, fc_weight2 = fc_bn_drop_layer(fc1_link, 256, train_phase, keep_prob, "fc2")

        # score layer
        y_pred, score_weight = score_layer(fc2, 3)

        parameters = (filter1, filter2, filter3, fc_weight1, fc_weight2, score_weight)

    return y_pred, parameters


def corr_num_acc(labels, logits):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return correct_num, accuracy


def loss(labels, logits, reg=None, parameters=None):
    with tf.variable_scope("loss"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy'))

        if parameters == None:
            cost = cross_entropy
        else:
            reg_loss = 0.0
            for para in parameters:
                reg_loss += reg * 0.5 * tf.nn.l2_loss(para)
            cost = cross_entropy + reg_loss
    return cost


# to do the evaluation part for the whole data
# not use all data together, but many batchs
def do_eval(sess, X_dataset, para_dataset, y_dataset, batch_size, correct_num, placeholders, merged=None, test_writer=None,
            global_step=None):
    # get the placeholders
    input_x, para_pl, input_y, train_phase, keep_prob = placeholders
    # calculate the epoch and rest data
    num_epoch = X_dataset.shape[0] // batch_size
    rest_data_size = X_dataset.shape[0] % batch_size

    index = 0
    count = 0.0
    indexs = np.arange(X_dataset.shape[0])

    for step in xrange(num_epoch):
        index, data, _ = sequence_get_data(X_dataset, para_dataset, y_dataset, indexs, index, batch_size)

        feed_dict = {input_x: data['X'], para_pl:data['p'], input_y: data['y'], train_phase: False, keep_prob: 1.0}

        if step != num_epoch - 1 or merged == None:
            num = sess.run(correct_num, feed_dict=feed_dict)
        else:
            summary, num = sess.run([merged, correct_num], feed_dict=feed_dict)
            # test_writer.add_summary(summary, global_step)

        count += num

    if rest_data_size != 0:
        # the rest data
        index, data, _ = sequence_get_data(X_dataset, y_dataset, indexs, index, rest_data_size)

        feed_dict = {input_x: data['X'], para_pl:data['p'], input_y: data['y'], train_phase: False, keep_prob: 1.0}

        num = sess.run(correct_num, feed_dict=feed_dict)

        count += num
    return count / X_dataset.shape[0]


def do_train_file(sess, placeholders, dir, train_file, SPAN, max_step, batch_size, keep_prob_v):
    input_x, para_pl, input_y, train_phase, keep_prob, train_step, loss_value, accuracy = placeholders

    X_train, para_train, y_train = prepare_dataset(dir, train_file, SPAN)

    indexs = get_random_seq_indexs(X_train)
    out_of_dataset = False
    last_index = 0

    loop_loss_v = 0.0
    loop_acc = 0.0

    # one loop, namely, one file
    for step in xrange(max_step):

        # should not happen
        if out_of_dataset == True:
            print "out of dataset"
            indexs = get_random_seq_indexs(X_train)
            last_index = 0
            out_of_dataset = False

        last_index, data, out_of_dataset = sequence_get_data(X_train, para_train, y_train, indexs, last_index,
                                                             batch_size)

        feed_dict = {input_x: data['X'], para_pl:data['p'], input_y: data['y'], train_phase: True, keep_prob: keep_prob_v}
        _, loss_v, acc = sess.run([train_step, loss_value, accuracy], feed_dict=feed_dict)

        loop_loss_v += loss_v
        loop_acc += acc

    loop_loss_v /= max_step
    loop_acc /= max_step
    return loop_loss_v, loop_acc

def time_show(before_time, last_loop_num, loop_now, total_loop, epoch_now, total_epoch, log, count = None, count_total = None):
    last_time = time.time()
    span_time = last_time - before_time
    rest_loop = total_loop - loop_now
    rest_epoch = total_epoch - epoch_now

    print ('last %d loop use %f minutes' % (last_loop_num, span_time * last_loop_num / 60))
    print ('rest loop need %.3f minutes' % (span_time * rest_loop / 60))
    print ('rest epoch need %.3f hours' % (span_time * rest_loop / 3600 + span_time * total_loop * rest_epoch / 3600))
    #for show cross valid total time
    if count != None:
        rest_count = count_total - count
        print ('rest total time need %.3f hours' % (span_time * rest_loop / 3600 + span_time * total_loop * rest_epoch / 3600 + span_time * total_loop * total_epoch * rest_count / 3600))

    log += ('last %d loop use %f minutes\n' % (last_loop_num, span_time * last_loop_num / 60))
    log += ('rest loop need %.3f minutes\n' % (span_time * rest_loop / 60))
    log += ('rest epoch need %.3f hours\n' % ((span_time * rest_loop / 3600) + (span_time * total_loop * rest_epoch /3600)))
    # for show cross valid total time
    if count != None:
        rest_count = count_total - count
        log += ('rest total time need %.3f hours\n' % (
        span_time * rest_loop / 3600 + span_time * total_loop * rest_epoch / 3600 + span_time * total_loop * total_epoch * rest_count / 3600))


def random_uniform_array(number, start, end):
    array = np.zeros(number)
    for i in np.arange(number - 2):
        array[i] = 10 ** np.random.uniform(start, end)
    array[-2] = 10 ** start
    array[-1] = 10 ** end

    return array
