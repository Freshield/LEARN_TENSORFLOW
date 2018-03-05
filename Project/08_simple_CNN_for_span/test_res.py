import tensorflow as tf
import numpy as np
import os


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

def num_to_one_hot(dataset, category_num):
    lines = dataset.shape[0]
    one_hot_dataset = np.zeros([lines,category_num], dtype=np.float32)
    one_hot_dataset[np.arange(lines), dataset] = 1
    return one_hot_dataset

def reshape_dataset(dataset, SPAN):
    input_data = np.zeros((dataset.shape[0], 32, 104, 3))
    temp_data = np.reshape(dataset[:, :6200], (dataset.shape[0], 31, 100, 2))
    input_data[:, :31, 2:102, 0] = temp_data[:, :, :, 0]#cause input size is 32 not 31
    input_data[:, :31, 2:102, 1] = temp_data[:, :, :, 1]
    input_data[:, :, :, 2] = np.reshape(np.tile(dataset[:, 6200:6241], 82)[:, :3328], (dataset.shape[0], 32, 104))

    output_data = dataset[:, 6240 + SPAN[0]]
    output_data = num_to_one_hot(output_data, 3)

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

#for create the pooling
def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#for create batch norm layer
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

#conv-bn-relu-maxpooling
def conv_bn_pool_layer(input_layer, filter_depth, train_phase, name):
    input_depth = input_layer.shape[-1]
    with tf.variable_scope(name):
        filter = weight_variable([3,3,input_depth,filter_depth], "filter")
        biases = bias_variable([filter_depth])
        conv_output = conv2d(input_layer, filter, 1, "SAME") + biases
        bn_output = batch_norm_layer(conv_output, train_phase, "conv_bn")
        act_output = tf.nn.relu(bn_output)
        output = max_pool_2x2(act_output)
    return output,filter

#fully connected layer
#fc-bn-relu-drop
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

#score layer
def score_layer(input_layer, label_size):
    with tf.variable_scope("score"):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, label_size], 'score_weight')
        b = bias_variable([label_size])
        output = tf.matmul(input_layer, W) + b
    return output, W

input_layer = tf.ones([100,32,104,3])

train_phase = tf.constant(True)

keep_prob = tf.constant(1.0)

#input (N,32,104,3)
bn_input = batch_norm_layer(input_layer, train_phase, "bn_input")

# conv1 (N,16,52,64)
conv1, filter1 = conv_bn_pool_layer(bn_input, 64, train_phase, "conv1")

# conv2 (N,8,26,128)
conv2, filter2 = conv_bn_pool_layer(conv1, 128, train_phase, "conv2")

# conv3 (N, 4, 13, 256)
conv3, filter3 = conv_bn_pool_layer(conv2, 256, train_phase, "conv3")

# flat
flat_conv3 = tf.reshape(conv3, [-1, 4 * 13 * 256])

# fc layer
fc1, fc_weight = fc_bn_drop_layer(flat_conv3, 1024, train_phase, keep_prob, "fc1")

# score layer
y_pred, score_weight = score_layer(fc1, 3)

parameters = (filter1, filter2, filter3, fc_weight, score_weight)
