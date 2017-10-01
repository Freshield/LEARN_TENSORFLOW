import basic_model as bm
import data_process_model as dpm
import numpy as np
import tensorflow as tf


#reshape the dataset for CNN
#Here is just an example
#ver 1.0
def reshape_dataset(dataset, SPAN):
    """
    #You need fill as your program

    input_data = np.zeros((dataset.shape[0], 32, 104, 2))
    temp_data = np.reshape(dataset[:, :6200], (-1, 31, 100, 2))
    input_data[:, :31, 2:102, 0] = temp_data[:, :, :, 0]  # cause input size is 32 not 31
    input_data[:, :31, 2:102, 1] = temp_data[:, :, :, 1]
    para_data = dataset[:, 6200:6241]

    output_data = dataset[:, 6240 + SPAN[0]].astype(int)
    output_data = num_to_one_hot(output_data, 3)

    return input_data, para_data, output_data
    """


#get the y_pred, define the whole net
#ver 1.0
def inference(input_layer, para_data, train_phase, keep_prob):
    """
    #The most important part for the network is
    #getting inference to return y_pred

    parameters = []
    #input shape should be (N,32,104,2)
    input_depth = input_layer.shape[-1]



    return y_pred, parameters
    """


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
