import basic_model as bm
import data_process_model as dpm
import numpy as np
import tensorflow as tf


#reshape the dataset for CNN
#Here is just an example
#ver 1.0
def reshape_dataset(dataset, SPAN):

    #You need fill as your program

    input_data = np.zeros((dataset.shape[0], 304, 104, 2))
    temp_data = np.reshape(dataset[:, :6200], (-1, 31, 100, 2))
    input_data[:, :31, 2:102, 0] = temp_data[:, :, :, 0]  # cause input size is 32 not 31
    input_data[:, :31, 2:102, 1] = temp_data[:, :, :, 1]
    para_data = dataset[:, 6200:6241]

    output_data = dataset[:, 6240 + SPAN[0]].astype(int)
    output_data = dpm.num_to_one_hot(output_data, 3)

    return input_data, para_data, output_data



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
#          556
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
        bn_input = bm.batch_norm_layer(input_layer, train_phase, 'bn_input')
        filter = bm.weight_variable([3,3,input_depth,64], 'filter_input')
        biases = bm.bias_variable([64])
        conv_input = bm.conv2d(bn_input, filter, 1, 'SAME') + biases

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
    fc1, p_fc = bm.fc_layer(avg_pool_flat, 512)
    parameters[0:0] = p_fc

    # link the para_data(N,556)
    fc1_link = tf.concat([fc1, para_data], axis=1)

    # fc layer2(N,256)
    fc2, fc_weight2 = bm.fc_bn_drop_layer(fc1_link, 256, train_phase, keep_prob, "fc2")
    parameters[0:0] = fc_weight2

    # score layer
    y_pred, score_weight = bm.score_layer(fc2, 3)
    parameters[0:0] = score_weight

    return y_pred, parameters
