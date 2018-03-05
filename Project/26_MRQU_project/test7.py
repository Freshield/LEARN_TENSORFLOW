import tensorflow as tf
import numpy as np

a = tf.constant([[1,0],[0,1],[1,0],[0,1]])

b = tf.argmax(a,axis=1)

sess = tf.InteractiveSession()

print a.eval()
print b.eval()