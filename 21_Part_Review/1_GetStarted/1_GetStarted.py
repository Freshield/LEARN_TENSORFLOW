import tensorflow as tf

rank1 = tf.constant([1.,2.,3.])

sess = tf.InteractiveSession()

print(rank1.eval())

rank2 = tf.constant([[1.,2.,3.],[4.,5.,6.]])

print(rank2.eval())

rank3 = tf.constant([[[1.,2.,3.]],[[7.,8.,9.]]])

print(rank3.eval())