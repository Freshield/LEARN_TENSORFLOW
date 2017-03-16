import tensorflow as tf

IMAGE_SIZE = 28

IMAGE_PIXES = 28 * 28

NUM_CLASSES = 10

def inference(images, hidden1_num, hidden2_num):
    with tf.name_scope('hidden1'):
        hidden1_weight = tf.Variable(tf.truncated_normal([IMAGE_PIXES, hidden1_num], stddev= 1. / tf.sqrt(float(IMAGE_PIXES))), name='hidden1_weight')
        hidden1_bias = tf.Variable(tf.zeros([hidden1_num]), name='hidden1_bias')
        hidden1_out = tf.nn.relu(tf.matmul(images, hidden1_weight) + hidden1_bias)
    with tf.name_scope('hidden2'):
        hidden2_weight = tf.Variable(tf.truncated_normal([hidden1_num, hidden2_num], stddev= 1. / tf.sqrt(float(hidden1_num))), name='hidden2_weight')
        hidden2_bias = tf.Variable(tf.zeros([hidden2_num]), name='hidden2_bias')
        hidden2_out = tf.nn.relu(tf.matmul(hidden1_out, hidden2_weight) + hidden2_bias)
    with tf.name_scope('linear'):
        linear_weight = tf.Variable(tf.truncated_normal([hidden2_num, NUM_CLASSES], stddev= 1. / tf.sqrt(float(hidden2_num))), name='linear_weight')
        linear_bias = tf.Variable(tf.zeros([NUM_CLASSES]), name='linear_bias')
        logits = tf.matmul(hidden2_out, linear_weight) + linear_bias

    inf_cache = (hidden1_weight, hidden2_weight)
    return logits, inf_cache

def loss(logits, lables, inf_cache, reg):
    hidden1_weight, hidden2_weight = inf_cache
    lables = tf.to_int64(lables)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lables, logits=logits, name='xentropy'), name='xentropy_mean')
    final_loss = cross_entropy + 0.5 * reg * tf.reduce_sum(hidden1_weight **2) + 0.5 * reg * tf.reduce_sum(hidden2_weight ** 2)
    return final_loss

def training(loss, learning_rate):

    #train_op = tf.train.GradientDescentOptimizer(learning_rate,name='gsd').minimize(loss,name='minimize')
    train_op = tf.train.AdamOptimizer(learning_rate, name='adam').minimize(loss, name='minimize')
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    correct_count = tf.reduce_sum(tf.cast(correct, tf.float32), name='correct_count')
    return correct_count

def placeholder_inputs(batch_size):
    images_pl = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXES), name='image_pl')
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size), name='label_pl')

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

