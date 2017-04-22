import tensorflow as tf
import numpy as np

a = tf.constant([0,1,2,3,4,5])

sess = tf.InteractiveSession()

print a.eval()

labels = np.eye(10)[[0,1,2,3,4,5]]
print labels

tf_labels = tf.eye(10)[[0,1,2,3,4,5]]