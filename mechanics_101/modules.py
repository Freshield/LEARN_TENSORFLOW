import tensorflow as tf

IMAGE_SIZE = 28

IMAGE_PIXES = 28 * 28

NUM_CLASSES = 10

def inference(images, hidden1_num, hidden2_num):
    hidden1_weight = tf.Variable(tf.truncated_normal([IMAGE_PIXES, hidden1_num], stddev= 1. / tf.sqrt(float(IMAGE_PIXES))))
    hidden1_bias = tf.Variable(tf.zeros([hidden1_num]))

    hidden2_weight = tf.Variable(tf.truncated_normal([hidden1_num, hidden2_num], stddev= 1. / tf.sqrt(float(hidden1_num))))
    hidden2_bias = tf.Variable(tf.zeros([hidden2_num]))

    linear_weight = tf.Variable(tf.truncated_normal([hidden2_num, NUM_CLASSES], stddev= 1. / tf.sqrt(float(hidden2_num))))
    linear_bias = tf.Variable(tf.zeros([NUM_CLASSES]))



    hidden1_out = tf.nn.relu(tf.matmul(images, hidden1_weight) + hidden1_bias)
    hidden2_out = tf.nn.relu(tf.matmul(hidden1_out, hidden2_weight) + hidden2_bias)
    logits = tf.matmul(hidden2_out, linear_weight) + linear_bias

    return logits

def loss(logits, lables):
    print ('logits,lables shape:',logits.shape, lables.shape)
    lables = tf.to_int64(lables)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lables, logits=logits))
    return cross_entropy

def training(loss, learning_rate):
    #for tensorboard
    tf.summary.scalar('loss',loss)

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    correct_count = tf.reduce_sum(tf.cast(correct, tf.float32))
    return correct_count

def placeholder_inputs(batch_size):
    images_pl = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXES))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))

    return images_pl, labels_pl

def fill_feed_dict(data_set, images_pl, labels_pl, batch_size):
    image_batch, label_batch = data_set.next_batch(batch_size)

    feed_dict = {
        images_pl : image_batch,
        labels_pl : label_batch
    }

    return feed_dict

def do_eval(sess, eval_correct, images_pl, labels_pl, data_set, batch_size):
    correct_count = 0
    num_per_epoch = data_set.num_examples // batch_size
    num_examples = num_per_epoch * batch_size

    for step in xrange(num_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_pl, labels_pl, batch_size)
        step_correct = sess.run(eval_correct, feed_dict=feed_dict)
        correct_count += step_correct

    precision = float(correct_count) / num_examples
    print ('num examples: %d num correct: %d precision : %0.04f' % (num_examples, correct_count, precision))

