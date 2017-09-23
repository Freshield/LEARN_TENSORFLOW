from basic_model import *
from file_system_model import *
import otto_resnet_simple_model_2 as model
import pandas as pd
import numpy as np
#the parameter need fill
#######################################################
#from network_model_example import *
test_size = 144368

batch_size = 100
test_filename = 'data/result_trapha_T.csv'
########################################################


#######################################################################
################      HELPER    #######################################
#######################################################################

def sequence_get_data(dataset, indexs, last_index, batch_size):
    next_index = last_index + batch_size

    if next_index > dataset.shape[0]:
        span_index = indexs[last_index:]
        out_of_dataset = True
    else:
        span_index = indexs[last_index:next_index]
        out_of_dataset = False

    data = dataset[span_index]
    return (next_index, data, out_of_dataset)


################################################################################

# change the dir and log dir nameodule_dir = module_dir + model_name + '/'

test_loops = test_size // batch_size
test_rest = test_size % batch_size
test_loop = 0

result_dataset = np.zeros((test_size,10), dtype=np.float32)
id_col = np.arange(test_size)
id_col += 1
result_dataset[:,0] = id_col

with tf.Graph().as_default():
    with tf.Session() as sess:
        # inputs
        input_data = tf.placeholder(tf.float32, [None, 9], name='input_data')

        norm = tf.nn.softmax(input_data)

        the_argmax = tf.argmax(norm, axis=1)

        one_hot = tf.one_hot(the_argmax, depth=9, dtype=tf.int32)

        # read file
        test_data = pd.read_csv(test_filename, header=None).values

        # do the test
        indexs = np.arange(test_size)

        res_last_index = 0
        res_next_index = 0

        while test_loop < test_loops:
            print '%d ' % test_loop,

            res_next_index = res_last_index + batch_size

            data = test_data[res_last_index:res_next_index,1:]

            feed_dict = {input_data:data}

            norm_v = sess.run(norm, feed_dict=feed_dict)

            result_dataset[res_last_index:res_next_index,1:] = norm_v

            res_last_index = res_next_index

            test_loop += 1

        if test_rest != 0:
            data = test_data[res_last_index:,1:]

            feed_dict = {input_data:data}

            norm_v = sess.run(norm, feed_dict=feed_dict)

            result_dataset[res_last_index:, 1:] = norm_v


        header = "id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9"

        fmt = '%i,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f'

        np.savetxt('data/result_trapha_T_oh.csv', result_dataset, delimiter=',', header=header, comments='', fmt=fmt)




