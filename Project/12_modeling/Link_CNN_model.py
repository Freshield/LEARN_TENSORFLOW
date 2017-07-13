from basic_model import *

#reshape the dataset for Link_CNN
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


#read the dataset from file
#needn't changed, just reply on reshape_dataset
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
