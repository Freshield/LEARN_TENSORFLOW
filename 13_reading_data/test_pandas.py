import pandas as pd
import tensorflow as tf
import numpy as np

filename = 'iris_test.csv'

batch_size = 5

data = pd.read_csv(filename, header=None)

lines_num = data.shape[0] - 1

def get_data():
    random_index = np.random.randint(lines_num, size=[batch_size])
    return data.values[random_index,:]

sess = tf.InteractiveSession()

input = tf.placeholder(tf.float32, [None, 5])

print sess.run(input, feed_dict={input:get_data()})