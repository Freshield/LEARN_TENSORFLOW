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

    def conv2d (self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2 (self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def mnist_model(self, wb_path, input_t, trainable=False):

        wb = Weight_Bag()
        wb.hdf5_to_restore_dic(wb_path)

        if trainable:
            get_tensor_op = wb.get_v_tensor
        else:
            get_tensor_op = wb.get_c_tensor

        W_conv1 = get_tensor_op('W_conv1')
        b_conv1 = get_tensor_op('b_conv1')
        h_conv1 = tf.nn.relu(self.conv2d(input_t,W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        W_conv2 = get_tensor_op(name='W_conv2')
        b_conv2 = get_tensor_op('b_conv2')
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        W_fc1 = get_tensor_op('W_fc1')
        b_fc1 = get_tensor_op('b_fc1')
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        W_fc2 = get_tensor_op('W_fc2')
        b_fc2 = get_tensor_op('b_fc2')
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
        return y_conv

    def multi_mnist_model(self,wb_dir,input_t,num_model,trainable=False):

        model_list = []

        for i in range(num_model):
            y = self.mnist_model(wb_dir+'mnist_conv_wb_%d.hdf5' % i,input_t,trainable)
            y = tf.reshape(y,(-1,10,1))
            model_list.append(y)

        mix_y = tf.concat(model_list,axis=2)

        return mix_y



def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_vaibale(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)



mnist = input_data.read_data_sets('../../data/mnist', one_hot=True)

save_path = '../../data/mnist/model/'
if not os.path.exists(save_path):
    tf.gfile.MakeDirs(save_path)

sess = tf.InteractiveSession()
model = Mnist_module()
num_model = 5

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

mix_y = model.multi_mnist_model('weights/',x_image,num_model)

mix_y_flat = tf.reshape(mix_y,(-1,10*num_model))

W = weight_variable([10*num_model,10],'W')
b = bias_vaibale([10],'b')
y = tf.matmul(mix_y_flat,W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()

for i in range(10000):
    if i % 50 == 0:
        print(i)
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels}))
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

