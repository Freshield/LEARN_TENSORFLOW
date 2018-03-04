import basic_model as bm
import data_process_model as dpm
import numpy as np
import tensorflow as tf


#reshape the dataset for CNN
#Here is just an example
#ver 1.0
def reshape_dataset(dataset):

    #You need fill as your program]

    #feature data (x,35)
    feature_data = dataset[:,:-1]

    #temp data (x,35,35,35)
    temp_data = triple_size_data(feature_data)
    #x,y (batch,35)
    x, y, _, _ = temp_data.shape

    #input data (batch, 40,40,35)
    input_data = np.zeros((x, y+5, y+5, y))

    input_data[:,2:-3,2:-3,:] = temp_data[:,:,:]

    output_data = dataset[:, -1].astype(int)
    output_data = dpm.num_to_one_hot(output_data, 2)

    return input_data, output_data

def triple_size_data(dataset):
    #a_x is 2
    #a_y is 3
    a_x, a_y = dataset.shape

    output_dataset = np.zeros((a_x, a_y, a_y, a_y))

    for batch_num in range(a_x):

        # to create 3x3 matrix

        raw_data = np.zeros((a_y, a_y))

        for j in xrange(a_y):
            right = dataset[batch_num, :j]
            left = dataset[batch_num, j:]
            raw_data[j, :] = np.concatenate((left, right))

        #print raw_data

        # to create 3x3x3 matrix

        result = np.zeros((a_y, a_y, a_y))

        first = raw_data[0]
        #first is 123;231;312;
        #print first

        for x in range(a_y):
            temp_data = np.zeros((a_y, a_y))
            temp_data[:, :] = raw_data[:, :]
            #exchange xth and first value
            temp_data[x] = first
            temp_data[0] = raw_data[x]
            result[x, :, :] = temp_data[:, :]

        output_dataset[batch_num, :, :, :] = result[:, :, :]

    return output_dataset



#bn->relu->conv
#the size is same
#ver 1.0
def bn_relu_conv_same_layer(input_layer, filter_size, filter_depth, train_phase, name):
    with tf.variable_scope(name):
        with tf.variable_scope('brc_s'):
            input_depth = input_layer.shape[-1]
            bn_layer = bm.batch_norm_layer(input_layer, train_phase, 'bn')
            relu_layer = tf.nn.relu(bn_layer, 'relu')
            filter = bm.weight_variable([filter_size, filter_size, input_depth, filter_depth], 'filter')
            biases = bm.bias_variable([filter_depth])
            conv_layer = bm.conv2d(relu_layer, filter, 1, 'SAME') + biases
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
            bn_layer = bm.batch_norm_layer(input_layer, train_phase, 'bn')
            relu_layer = tf.nn.relu(bn_layer, 'relu')
            filter = bm.weight_variable([filter_size, filter_size, input_depth, filter_depth], 'filter')
            biases = bm.bias_variable([filter_depth])
            conv_layer = bm.conv2d(relu_layer, filter, 2, 'VALID') + biases
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

#get the y_pred, define the whole net
#architecture:
#
#   40*40*35
#       |
#   40*40*128
#       |
#   20*20*128
#       |
#   10*10*256
#       |
#    5*5*512
#       |
#   avg pooling
#       |
#   1*1*512
#       |
#     flat
#       |
#       2
#ver 1.0
def inference(input_layer, train_phase, keep_prob):
    parameters = []
    #input shape should be (N,40,40,35)
    input_depth = input_layer.shape[-1]

    #input[N,40,40,35],output[N,40,40,128]
    with tf.variable_scope('preprocess'):
        bn_input = bm.batch_norm_layer(input_layer, train_phase, 'bn_input')
        filter = bm.weight_variable([3,3,input_depth,128], 'filter_input')
        biases = bm.bias_variable([128])
        conv_input = bm.conv2d(bn_input, filter, 1, 'SAME') + biases

    #input[N,40,40,128],output[N,20,20,128]
    #first layer not change the depth
    resnet_l1, p1 = resnet_layer(conv_input, 128, train_phase, 'resnet_l1')
    parameters[0:0] = p1

    #input[N,20,20,128],output[N,10,10,256]
    resnet_l2, p2 = resnet_layer(resnet_l1, 256, train_phase, 'resnet_l2')
    parameters[0:0] = p2

    #input[N,10,20,256],output[N,5,5,512]
    resnet_l3, p3 = resnet_layer(resnet_l2, 512, train_phase, 'resnet_l3')
    parameters[0:0] = p3

    #input[N,5,5,512],output[N,1,1,512]
    avg_pool_layer = tf.nn.avg_pool(resnet_l3, [1,5,5,1], [1,1,1,1], 'VALID')

    #platten for fc
    #input[N,5,5,512],output[N,1*1*512]
    avg_pool_flat = tf.reshape(avg_pool_layer, [-1, 512])

    # score layer
    y_pred, score_weight = bm.score_layer(avg_pool_flat, 2)
    parameters[0:0] = score_weight

    return y_pred, parameters
