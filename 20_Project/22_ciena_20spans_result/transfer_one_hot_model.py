from file_system_model import *
import pandas as pd
import numpy as np
#the parameter need fill
#######################################################
#from network_model_example import *
test_size = 100000

batch_size = 100
test_filename = 'result_value.csv'
########################################################


# change the dir and log dir nameodule_dir = module_dir + model_name + '/'

test_loops = test_size // batch_size
test_rest = test_size % batch_size
test_loop = 0

result_dataset = np.zeros((test_size,6), dtype=np.float32)

with tf.Graph().as_default():
    with tf.Session() as sess:
        # inputs
        input_data = tf.placeholder(tf.float32, [None, 6], name='input_data')

        norm = tf.nn.softmax(input_data)

        # read file
        test_data = pd.read_csv(test_filename, header=None).values

        res_last_index = 0
        res_next_index = 0

        while test_loop < test_loops:
            print '%d ' % test_loop,

            res_next_index = res_last_index + batch_size

            data = test_data[res_last_index:res_next_index,:]

            feed_dict = {input_data:data}

            norm_v = sess.run(norm, feed_dict=feed_dict)

            result_dataset[res_last_index:res_next_index,:] = norm_v

            res_last_index = res_next_index

            test_loop += 1

        if test_rest != 0:
            data = test_data[res_last_index:,:]

            feed_dict = {input_data:data}

            norm_v = sess.run(norm, feed_dict=feed_dict)

            result_dataset[res_last_index:, :] = norm_v

        np.savetxt('result_probability.csv', result_dataset, delimiter=',')




