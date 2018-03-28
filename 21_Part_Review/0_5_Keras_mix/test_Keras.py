import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

from keras.layers import Dense, BatchNormalization
from keras.metrics import categorical_accuracy
print(K.learning_phase())

# build module

img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

x = Dense(128, activation='relu')(img)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
prediction = Dense(10, activation='softmax')(x)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=labels))

train_optim = tf.train.AdamOptimizer().minimize(loss)

mnist_data = input_data.read_data_sets('MNIST_data/', one_hot=True)

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(5000):
        batch_x, batch_y = mnist_data.train.next_batch(50)
        sess.run(train_optim, feed_dict={img: batch_x, labels: batch_y, K.learning_phase():True})
        if i % 100 == 0:
            print(i)

    acc_pred = categorical_accuracy(labels, prediction)
    pred = sess.run(acc_pred, feed_dict={labels: mnist_data.test.labels, img: mnist_data.test.images,
                                         K.learning_phase():False})

    print('accuracy: %.3f' % (sum(pred) / len(mnist_data.test.labels)))