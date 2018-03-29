from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import shutil
import h5py

class Weight_Bag:

    restore_dic = {}
    save_dic = {}

    def get_v_tensor(self,name):
        tensor = tf.Variable(self.restore_dic[name],name=name)
        return tensor

    def get_c_tensor(self,name):
        tensor = tf.constant(self.restore_dic[name],name=name)
        return tensor

    def save_tensor(self,sess,tensor,name):
        self.save_dic[name] = sess.run(tensor)

    def save_dic_to_hdf5(self,name):
        with h5py.File(name,'w') as f:
            for key,value in self.save_dic.items():
                f.create_dataset(key,data=value,compression='gzip')

    def hdf5_to_restore_dic(self,name):
        with h5py.File(name,'r') as f:
            for key in f.keys():
                self.restore_dic[key] = f[key].value

mnist = input_data.read_data_sets('../../data/mnist', one_hot=True)

for i in range(100):
    print(i)

    save_path = '../../data/mnist/model/'
    if not os.path.exists(save_path):
        tf.gfile.MakeDirs(save_path)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]), name='W')
    b = tf.Variable(tf.zeros([10]), name='b')
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    )

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()

    tf.global_variables_initializer().run()

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

    saver.save(sess, save_path + 'model.ckpt')
    print('done save model')

    wb = Weight_Bag()
    wb.save_tensor(sess, W, 'W')
    wb.save_tensor(sess, b, 'b')

    wb.save_dic_to_hdf5('mnist_wb_%d.hdf5' % i)
