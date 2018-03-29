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

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_vaibale(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
save_path = '../../data/mnist/model/'
if not os.path.exists(save_path):
    tf.gfile.MakeDirs(save_path)

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x,[-1,28,28,1])

W_conv1 = weight_variable([5,5,1,32],'W_conv1')
b_conv1 = bias_vaibale([32],'b_conv1')
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64],name='W_conv2')
b_conv2 = bias_vaibale([64],'b_conv2')
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024],'W_fc1')
b_fc1 = bias_vaibale([1024],'b_fc1')
h_pool2_flat = tf.reshape(h_pool2,[-1,7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024,10],'W_fc2')
b_fc2 = bias_vaibale([10],'b_fc2')
y_conv = tf.matmul(h_fc1,W_fc2) + b_fc2

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()

saver = tf.train.Saver()

tf.global_variables_initializer().run()

for i in range(1000):
    if i % 50 == 0:
        print(i)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
'''
saver.save(sess,save_path+'model.ckpt')
print('done save model')


wb = Weight_Bag()
wb.save_tensor(sess,W,'W')
wb.save_tensor(sess,b,'b')


print(wb.save_dic)
wb.save_dic_to_hdf5('mnist_wb_4.hdf5')

wb.hdf5_to_restore_dic('mnist_wb_4.hdf5')
print(wb.restore_dic)
'''