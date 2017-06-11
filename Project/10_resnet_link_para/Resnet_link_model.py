import tensorflow as tf
import numpy as np
import os
import pandas as pd
import time

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
    lines_num = X_dataset.shape[0]
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
        span_index = indexs[np.concatenate((last_part, before_part))]#link two parts together
        out_of_dataset = True
    else:
        span_index = indexs[last_index:next_index]

    X_data = X_dataset[span_index]
    y_data = y_dataset[span_index]
    return (next_index, {'X':X_data,'y':y_data}, out_of_dataset)

#normalize the dataset in different parts
def normalize_dataset(dataset, min_values=None, max_values=None):
    norm_dataset = np.zeros((dataset.shape))
    norm_dataset[:, :] = dataset[:, :]

    if min_values == None:
        CMr_min = np.min(norm_dataset[:,0:3100])
        CMi_min = np.min(norm_dataset[:,3100:6200])
        CD_min = np.min(norm_dataset[:,6200:6201])
        length_min = np.min(norm_dataset[:,6201:6221])
        power_min = np.min(norm_dataset[:,6221:6241])
    else:
        CMr_min, CMi_min, CD_min, length_min, power_min = min_values


    if max_values == None:
        CMr_max = np.max(norm_dataset[:,0:3100])
        CMi_max = np.max(norm_dataset[:,3100:6200])
        CD_max = np.max(norm_dataset[:,6200:6201])
        length_max = np.max(norm_dataset[:,6201:6221])
        power_max = np.max(norm_dataset[:,6221:6241])
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

#reshape the dataset into 32x100x3
def reshape_dataset(dataset, SPAN):
    input_data = np.zeros((dataset.shape[0], 32, 100, 3))
    temp_data = np.reshape(dataset[:, :6200], (dataset.shape[0], 31, 100, 2))
    input_data[:, :31, :, 0] = temp_data[:, :, :, 0]
    input_data[:, :31, :, 1] = temp_data[:, :, :, 1]
    input_data[:, :, :, 2] = np.reshape(np.tile(dataset[:, 6200:6241], 79)[:, :3200], (dataset.shape[0], 32, 100))

    output_data = dataset[:, 6240 + SPAN[0]]
    output_data = np_utils.to_categorical(output_data)

    return input_data, output_data

###########################################################
################# graph helper ############################
###########################################################

#create weights
def weight_variable(shape, name):
  """weight_variable generates a weight variable of a given shape."""
  weight = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

  return weight


#create biases
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

#for create convolution kernel
def conv2d(x, W, stride, padding):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def batch_norm_layer(x, train_phase, scope_bn):
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = np.arange(len(x.shape) - 1)
        batch_mean, batch_var = tf                          .nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

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


def resnet_diffD_halfS_block(input_layer, block_depth, train_phase):
    input_depth = input_layer.shape[-1]
    #check
    if input_depth < 4:
        raise TypeError('input depth is too small')
    small_depth = block_depth // 4
    with tf.variable_scope('res_dhb'):
        block_layer1, f1 = bn_relu_conv_half_layer(input_layer, 1, small_depth, train_phase, 'bl_1')
        block_layer2, f2 = bn_relu_conv_same_layer(block_layer1, 3, small_depth, train_phase, 'bl_2')
        block_layer3, f3 = bn_relu_conv_same_layer(block_layer2, 1, block_depth, train_phase, 'bl_3')
        block_layer4, f4 = bn_relu_conv_half_layer(input_layer, 1, block_depth, train_phase, 'bl_4')
        add_layer = block_layer4 + block_layer3
        parameters = (f1, f2, f3, f4)
    return add_layer, parameters

def resnet_first_layer(input_layer, layer_depth, train_phase, name):
    parameters = []
    with tf.variable_scope(name):
        with tf.variable_scope('res_1st'):
            layer1, p1 = resnet_diffD_sameS_block(input_layer, layer_depth, train_phase)
            layer2, p2 = resnet_same_block(layer1, train_phase)
    parameters[0:0] = p1
    parameters[0:0] = p2
    return layer2, parameters

def resnet_layer(input_layer, layer_depth, train_phase, name):
    parameters = []
    with tf.variable_scope(name):
        with tf.variable_scope('res'):
            layer1, p1 = resnet_diffD_halfS_block(input_layer, layer_depth, train_phase)
            layer2, p2 = resnet_same_block(layer1, train_phase)
    parameters[0:0] = p1
    parameters[0:0] = p2
    return layer2, parameters

def fc_layer(input_layer, label_size):
    with tf.variable_scope('fc'):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, label_size], 'fc_weight')
        b = bias_variable([label_size])
        output = tf.matmul(input_layer, W) + b
    return output, [W]

def inference(input_layer, train_phase):
    parameters = []
    #input shape should be (N,32,100,3)
    input_depth = input_layer.shape[-1]

    #input[N,32,100,3],output[N,32,100,64]
    with tf.variable_scope('preprocess'):
        bn_input = batch_norm_layer(input_layer, train_phase, 'bn_input')
        filter = weight_variable([3,3,input_depth,64], 'filter_input')
        biases = bias_variable([64])
        conv_input = conv2d(bn_input, filter, 1, 'SAME') + biases

    #input[N,32,100,64],output[N,32,100,256]
    #first layer not change the depth
    resnet_l1, p1 = resnet_first_layer(conv_input, 256, train_phase, 'resnet_l1')
    parameters[0:0] = p1

    #input[N,32,100,256],output[N,16,50,512]
    resnet_l2, p2 = resnet_layer(resnet_l1, 512, train_phase, 'resnet_l2')
    parameters[0:0] = p2

    #input[N,16,50,512],output[N,8,25,1024]
    resnet_l3, p3 = resnet_layer(resnet_l2, 1024, train_phase, 'resnet_l3')
    parameters[0:0] = p3

    #pad for avg pool
    #input[N,8,25,1024],output[N,12,27,1024]
    resnet_l3_pad = tf.pad(resnet_l3, [[0,0],[2,2],[1,1],[0,0]], 'CONSTANT', 'l3_pad')
    #input[N,12,27,1024],output[N,4,9,1024]
    avg_pool_layer = tf.nn.avg_pool(resnet_l3_pad, [1,3,3,1], [1,3,3,1], 'VALID')

    #platten for fc
    #input[N,12,27,1024],output[N,4*9*1024]
    avg_pool_flat = tf.reshape(avg_pool_layer, [-1, 4 * 9 * 1024])

    #fc layer
    #input[N,4*9*1024],output[N,3]
    y_pred, p_fc = fc_layer(avg_pool_flat, 3)
    parameters[0:0] = p_fc

    return y_pred, parameters

def corr_num_acc(labels, logits):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return correct_num, accuracy

#to do the evaluation part for the whole data
#not use all data together, but many batchs
def do_eval(sess, X_dataset, y_dataset, batch_size, correct_num, placeholders, merged=None, test_writer=None,
            global_step=None):

    input_x, input_y, train_phase = placeholders
    num_epoch = X_dataset.shape[0] // batch_size
    rest_data_size = X_dataset.shape[0] % batch_size

    index = 0
    count = 0.0
    indexs = np.arange(X_dataset.shape[0])

    for step in xrange(num_epoch):
        index, data, _ = sequence_get_data(X_dataset, y_dataset, indexs, index, batch_size)

        if step == num_epoch - 1:
            if merged != None :
                summary, num = sess.run([merged, correct_num], feed_dict={input_x:data['X'], input_y:data['y'],
                                                                          train_phase:False})
                #add summary
                #test_writer.add_summary(summary, global_step)
            else:
                num = sess.run(correct_num, feed_dict={input_x:data['X'], input_y:data['y'], train_phase:False})

        else:
            num = sess.run(correct_num, feed_dict={input_x:data['X'], input_y:data['y'],train_phase:False})

        count += num

    if rest_data_size != 0:
        #the rest data
        index, data, _ = sequence_get_data(X_dataset, y_dataset, indexs, index, rest_data_size)
        num = sess.run(correct_num, feed_dict={input_x:data['X'], input_y:data['y'], train_phase:False})

        count += num
    return count / X_dataset.shape[0]

def loss(labels, logits, reg=None, parameters=None):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits,name='xentropy'))

    if parameters == None:
        cost = cross_entropy
    else:
        reg_loss = 0.0
        for para in parameters:
            reg_loss += reg * 0.5 * tf.nn.l2_loss(para)
        cost = cross_entropy + reg_loss
    return cost
