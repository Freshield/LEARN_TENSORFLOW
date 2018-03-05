import tensorflow as tf
import numpy as np
import os
import pandas as pd
import time

############################################################
############# helpers ######################################
############################################################

#to copy files from source dir to target dir
#ver 1.0
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

# ensure the path exist
#will reset the dir
#ver 1.0
def del_and_create_dir(dir_path):
    if tf.gfile.Exists(dir_path):
        tf.gfile.DeleteRecursively(dir_path)
    tf.gfile.MakeDirs(dir_path)


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

#set num to one hot array
#ver 1.0
def num_to_one_hot(dataset, category_num):
    lines = dataset.shape[0]
    one_hot_dataset = np.zeros([lines, category_num], dtype=np.float32)
    one_hot_dataset[np.arange(lines), dataset] = 1
    return one_hot_dataset

#reshape the dataset for CNN
#ver 1.0
def reshape_dataset(dataset, SPAN):
    input_data = np.zeros((dataset.shape[0], 32, 104, 2))
    temp_data = np.reshape(dataset[:, :6200], (-1, 31, 100, 2))
    input_data[:, :31, 2:102, 0] = temp_data[:, :, :, 0]  # cause input size is 32 not 31
    input_data[:, :31, 2:102, 1] = temp_data[:, :, :, 1]
    para_data = dataset[:, 6200:6241]

    output_data = dataset[:, 6240 + SPAN[0]].astype(int)
    output_data = num_to_one_hot(output_data, 3)

    return input_data, para_data, output_data

#read the dataset from file
#ver 1.0
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

#create weights
#ver 1.0
def weight_variable(shape, name):
  """weight_variable generates a weight variable of a given shape."""
  weight = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

  return weight


#create biases
#ver 1.0
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

#for create convolution kernel
#ver 1.0
def conv2d(x, W, stride, padding):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


# for create batch norm layer
#ver 1.0
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

#bn->relu->conv
#the size is same
#ver 1.0
def bn_relu_conv_same_layer(input_layer, filter_size, filter_depth, train_phase, name):
    with tf.variable_scope(name):
        with tf.variable_scope('brc_s'):
            input_depth = input_layer.shape[-1]
            bn_layer = batch_norm_layer(input_layer, train_phase, 'bn')
            relu_layer = tf.nn.relu(bn_layer, 'relu')
            filter = weight_variable([filter_size, filter_size, input_depth, filter_depth], 'filter')
            biases = bias_variable([filter_depth])
            conv_layer = conv2d(relu_layer, filter, 1, 'SAME') + biases
    return conv_layer, filter

#bn->relu->conv
#size is half, namely, stride is 2
#care for the filter size, 1 or 2
#ver 1.0
def bn_relu_conv_half_layer(input_layer, filter_size, filter_depth, train_phase, name):
    #filter size check
    if filter_size != 1 and filter_size != 2:
        raise ValueError('filter size should be 1 or 2')
    with tf.variable_scope(name):
        with tf.variable_scope('brc_h'):
            input_depth = input_layer.shape[-1]
            bn_layer = batch_norm_layer(input_layer, train_phase, 'bn')
            relu_layer = tf.nn.relu(bn_layer, 'relu')
            filter = weight_variable([filter_size, filter_size, input_depth, filter_depth], 'filter')
            biases = bias_variable([filter_depth])
            conv_layer = conv2d(relu_layer, filter, 2, 'VALID') + biases
    return conv_layer, filter

#resnet basic block
#input and output same depth and same size
#architecture:
#
#input
#|    \
#|    1*1*input_depth*small_depth
#|     |
#|    3*3*small_depth*small_depth
#|     |
#|    1*1*small_depth*input_depth
#|     |
#+-----
#|
#output
#ver 1.0
def resnet_same_block(input_layer, train_phase):
    input_depth = input_layer.shape[-1]
    #check
    if input_depth < 4:
        raise TypeError('input depth is too small')
    small_depth = input_depth // 4
    with tf.variable_scope('res_sb'):
        block_layer1, f1 = bn_relu_conv_same_layer(input_layer, 1, small_depth, train_phase, 'bl_1')
        block_layer2, f2 = bn_relu_conv_same_layer(block_layer1, 3, small_depth, train_phase, 'bl_2')
        block_layer3, f3 = bn_relu_conv_same_layer(block_layer2, 1, input_depth, train_phase, 'bl_3')
        add_layer = input_layer + block_layer3
        parameters = (f1, f2, f3)
    return add_layer, parameters

#resnet basic block
#input and output different depth and same size
#architecture:
#
#                  input
#                  |    \
#                  |    1*1*input_depth*small_depth
#1*1*input_depth*block_depth |
#                 |    3*3*small_depth*small_depth
#                 |     |
#                 |    1*1*small_depth*block_depth
#                 |     |
#                 +-----
#                 |
#               output
#ver 1.0
def resnet_diffD_sameS_block(input_layer, block_depth, train_phase):
    input_depth = input_layer.shape[-1]
    #check
    if input_depth < 4:
        raise TypeError('input depth is too small')
    small_depth = block_depth // 4
    with tf.variable_scope('res_dsb'):
        block_layer1, f1 = bn_relu_conv_same_layer(input_layer, 1, small_depth, train_phase, 'bl_1')
        block_layer2, f2 = bn_relu_conv_same_layer(block_layer1, 3, small_depth, train_phase, 'bl_2')
        block_layer3, f3 = bn_relu_conv_same_layer(block_layer2, 1, block_depth, train_phase, 'bl_3')
        block_layer4, f4 = bn_relu_conv_same_layer(input_layer, 1, block_depth, train_phase, 'bl_4')
        add_layer = block_layer4 + block_layer3
        parameters = (f1, f2, f3, f4)
    return add_layer, parameters


#resnet basic block
#input and output different depth and half size
#architecture:
#
#                      input
#                      |    \
#                      |    2*2*input_depth*small_depth(stride=2)
#2*2*input_depth*block_depth(stride=2) |
#                      |    3*3*small_depth*small_depth
#                      |     |
#                      |    1*1*small_depth*block_depth
#                      |     |
#                      +-----
#                      |
#                     output
#ver 1.0
def resnet_diffD_halfS_block(input_layer, block_depth, train_phase):
    input_depth = input_layer.shape[-1]
    #check
    if input_depth < 4:
        raise TypeError('input depth is too small')
    small_depth = block_depth // 4
    with tf.variable_scope('res_dhb'):
        block_layer1, f1 = bn_relu_conv_half_layer(input_layer, 2, small_depth, train_phase, 'bl_1')
        block_layer2, f2 = bn_relu_conv_same_layer(block_layer1, 3, small_depth, train_phase, 'bl_2')
        block_layer3, f3 = bn_relu_conv_same_layer(block_layer2, 1, block_depth, train_phase, 'bl_3')
        block_layer4, f4 = bn_relu_conv_half_layer(input_layer, 2, block_depth, train_phase, 'bl_4')
        add_layer = block_layer4 + block_layer3
        parameters = (f1, f2, f3, f4)
    return add_layer, parameters

#resnet first layer
#change the depth
#ver 1.0
def resnet_first_layer(input_layer, layer_depth, train_phase, name):
    parameters = []
    with tf.variable_scope(name):
        with tf.variable_scope('res_1st'):
            layer1, p1 = resnet_diffD_sameS_block(input_layer, layer_depth, train_phase)
            layer2, p2 = resnet_same_block(layer1, train_phase)
    parameters[0:0] = p1
    parameters[0:0] = p2
    return layer2, parameters

#resnet normal layer
#a half block with a same block
#output is half size and layer depth
#ver 1.0
def resnet_layer(input_layer, layer_depth, train_phase, name):
    parameters = []
    with tf.variable_scope(name):
        with tf.variable_scope('res'):
            layer1, p1 = resnet_diffD_halfS_block(input_layer, layer_depth, train_phase)
            layer2, p2 = resnet_same_block(layer1, train_phase)
    parameters[0:0] = p1
    parameters[0:0] = p2
    return layer2, parameters

#the fully connect layer
#ver 1.0
def fc_layer(input_layer, label_size):
    with tf.variable_scope('fc'):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, label_size], 'fc_weight')
        b = bias_variable([label_size])
        output = tf.matmul(input_layer, W) + b
    return output, [W]


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
    return output, [W]

# score layer
def score_layer(input_layer, label_size):
    with tf.variable_scope("score"):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, label_size], 'score_weight')
        b = bias_variable([label_size])
        output = tf.matmul(input_layer, W) + b
    return output, [W]

#get the y_pred, define the whole net
#architecture:
#
#   32*104*2
#       |
#   32*104*64
#       |
#   32*104*128
#       |
#   16*52*256
#       |
#   8*26*512
#       |
#   4*13*1024
#       |
#   avg pooling
#       |
#   2*5*1024
#       |
#     flat
#       |
#       fc
#       |
#       512 + 41
#           |
#          553
#           |
#          256
#           |
#           3
#ver 1.0
def inference(input_layer, para_data, train_phase, keep_prob):
    parameters = []
    #input shape should be (N,32,104,2)
    input_depth = input_layer.shape[-1]

    #input[N,32,100,3],output[N,32,104,64]
    with tf.variable_scope('preprocess'):
        bn_input = batch_norm_layer(input_layer, train_phase, 'bn_input')
        filter = weight_variable([3,3,input_depth,64], 'filter_input')
        biases = bias_variable([64])
        conv_input = conv2d(bn_input, filter, 1, 'SAME') + biases

    #input[N,32,104,64],output[N,32,104,128]
    #first layer not change the depth
    resnet_l1, p1 = resnet_first_layer(conv_input, 128, train_phase, 'resnet_l1')
    parameters[0:0] = p1

    #input[N,32,104,128],output[N,16,52,256]
    resnet_l2, p2 = resnet_layer(resnet_l1, 256, train_phase, 'resnet_l2')
    parameters[0:0] = p2

    #input[N,16,52,256],output[N,8,26,512]
    resnet_l3, p3 = resnet_layer(resnet_l2, 512, train_phase, 'resnet_l3')
    parameters[0:0] = p3

    #input[N,8,26,512],output[N,4,13,1024]
    resnet_l4, p4 = resnet_layer(resnet_l3, 1024, train_phase, 'resnet_l4')
    parameters[0:0] = p4

    #pad for avg pool
    #input[N,4,13,1024],output[N,6,15,1024]
    resnet_l4_pad = tf.pad(resnet_l4, [[0,0],[1,1],[1,1],[0,0]], 'CONSTANT', 'l4_pad')
    #input[N,6,15,1024],output[N,2,5,1024]
    avg_pool_layer = tf.nn.avg_pool(resnet_l4_pad, [1,3,3,1], [1,3,3,1], 'VALID')

    #platten for fc
    #input[N,2,5,1024],output[N,2*5*1024]
    avg_pool_flat = tf.reshape(avg_pool_layer, [-1, 2 * 5 * 1024])

    #fc layer
    #input[N,4*9*1024],output[N,512]
    fc1, p_fc = fc_layer(avg_pool_flat, 512)
    parameters[0:0] = p_fc

    # link the para_data(N,556)
    fc1_link = tf.concat([fc1, para_data], axis=1)

    # fc layer2(N,256)
    fc2, fc_weight2 = fc_bn_drop_layer(fc1_link, 256, train_phase, keep_prob, "fc2")
    parameters[0:0] = fc_weight2

    # score layer
    y_pred, score_weight = score_layer(fc2, 3)
    parameters[0:0] = score_weight

    return y_pred, parameters

#get the correct number and accuracy
#ver 1.0
def corr_num_acc(labels, logits):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return correct_num, accuracy

#get the loss
#ver 1.0
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
#ver 1.0
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

#train loop in one file
#ver 1.0
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

#to show the time
#ver 1.0
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

#to get the random hypers for cross valid
#ver 1.0
def random_uniform_array(number, start, end):
    array = np.zeros(number)
    for i in np.arange(number - 2):
        array[i] = 10 ** np.random.uniform(start, end)
    array[-2] = 10 ** start
    array[-1] = 10 ** end

    return array
