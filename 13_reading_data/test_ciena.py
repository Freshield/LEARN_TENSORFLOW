import pandas as pd
import tensorflow as tf
import numpy as np

#filename = 'iris_test.csv'

batch = 100
lr_rate = 0.01
max_step = 30000

train_dataset = pd.read_csv('ciena.csv', header=None)
#test_dataset = pd.read_csv('iris_test.csv', header=None)

print train_dataset.shape

def get_batch_data(data_set, batch_size):
    lines_num = data_set.shape[0] - 1
    random_index = np.random.randint(lines_num, size=[batch_size])
    features = data_set.values[random_index,:-1]
    labels = data_set.values[random_index,-1]
    np_labels = np.array(labels, dtype=np.int32)
    labels_one_hot = np.eye(3)[np_labels]
    return {'features':features, 'labels':labels_one_hot}

def get_whole_data(data_set):
    features = data_set.values[:,:-1]
    labels = data_set.values[:,-1]
    np_labels = np.array(labels, dtype=np.int32)
    labels_one_hot = np.eye(3)[np_labels]
    return {'features':features, 'labels':labels_one_hot}

print get_batch_data(train_dataset, 5)['labels'].shape

x = tf.placeholder(tf.float32, [None, 6260], name='input_x')
y_ = tf.placeholder(tf.float32, [None, 3], name='input_y')

W1 = tf.Variable(tf.truncated_normal([6260, 10000], stddev=1.0))
b1 = tf.Variable(tf.zeros([10000]))
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([10000, 3000], stddev=1.0))
b2 = tf.Variable(tf.zeros([3000]))
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

keep_prob = tf.placeholder(tf.float32)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([3000, 3], stddev=1.0))
b3 = tf.Variable(tf.zeros([3]))
y = tf.matmul(hidden2_drop, W3) + b3

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(lr_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


  # Train
for step in range(max_step):
    data = get_batch_data(train_dataset, batch)
    #data = get_whole_data(train_dataset)

    _, loss = sess.run([train_step,cross_entropy],
                       feed_dict={x: data['features'], y_: data['labels'], keep_prob:0.5})

    if step % 100 == 0:
        print 'loss in step %d is %f' % (step, loss)



    if step % 1000 == 0:
        data = get_batch_data(train_dataset, batch)
        result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob:1.0})
        print 'accuracy in step %d is %f' % (step, result)
  # Test trained model

data = get_whole_data(train_dataset)
result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob:1.0})
print 'last accuracy is %f' % (result)
""""""