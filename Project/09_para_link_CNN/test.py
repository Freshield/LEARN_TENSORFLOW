#import tensorflow as tf
import numpy as np
'''
a = tf.zeros([10,5])
b = tf.ones([10,2])

c = tf.concat([a,b], axis=1)

sess = tf.InteractiveSession()

print c.eval()
'''

def random_uniform_array(number, start, end):
    array = np.zeros(number)
    for i in np.arange(number - 2):
        array[i] = 10 ** np.random.uniform(start, end)
    array[-2] = 10 ** start
    array[-1] = 10 ** end

    return array

d = random_uniform_array(10, 0 ,1)

print d

e = 1
f = 3
print int(10 * (float(e) / float(f)))

