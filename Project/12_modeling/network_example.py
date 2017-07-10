import tensorflow as tf
import numpy as np

from data_process_model import *
from basic_model import *


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
