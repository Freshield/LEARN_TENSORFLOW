from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import numpy as np
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

class Mnist_module:

    def mnist_model(self, wb_path, input_t, trainable=False):

        wb = Weight_Bag()
        wb.hdf5_to_restore_dic(wb_path)

        if trainable:
            get_tensor_op = wb.get_v_tensor
        else:
            get_tensor_op = wb.get_c_tensor

        W = get_tensor_op('W')
        b = get_tensor_op('b')
        y = tf.matmul(input_t, W) + b
        return y

    def multi_mnist_model(self,wb_dir,input_t,num_model,trainable=False):

        model_list = []

        for i in range(num_model):
            y = self.mnist_model(wb_dir+'mnist_wb_%d.hdf5' % i,input_t,trainable)
            y = tf.reshape(y,(-1,10,1))
            model_list.append(y)

        y = self.mnist_model(wb_dir+'mnist_wb_0.hdf5', input_t, True)
        y = tf.reshape(y,(-1,10,1))
        model_list.append(y)

        mix_y = tf.concat(model_list,axis=2)

        return mix_y






mnist = input_data.read_data_sets('../../data/mnist', one_hot=True)

save_path = '../../data/mnist/model/'
if not os.path.exists(save_path):
    tf.gfile.MakeDirs(save_path)

sess = tf.InteractiveSession()
model = Mnist_module()
num_model = 100

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

mix_y = model.multi_mnist_model('',x,num_model)

mix_y_flat = tf.reshape(mix_y,(-1,10*(num_model+1)))

W = tf.Variable(tf.zeros([10*(num_model+1),10]))
y = tf.matmul(mix_y_flat,W)

tf.global_variables_initializer().run()

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(20000):
    if i % 100 == 0:
        print(i)
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels}))
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

