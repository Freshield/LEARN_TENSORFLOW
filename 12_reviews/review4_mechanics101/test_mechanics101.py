import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math
import time

batch_size = 100
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10
hidden1_size = 128
hidden2_size = 32
learning_rate = 0.01
max_step = 10000


data_sets = input_data.read_data_sets('MNIST_data')

if tf.gfile.Exists('logs'):
    tf.gfile.DeleteRecursively('logs')
tf.gfile.MakeDirs('logs')

def inference(images, hidden1_size, hidden2_size):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_size],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS)),
                                name='weights'))

        biases = tf.Variable(tf.zeros([hidden1_size]),
                             name='biases')

        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_size, hidden2_size],
                                stddev=1.0 / math.sqrt(float(hidden1_size)),
                                name='weights')
        )

        biases = tf.Variable(tf.zeros([hidden2_size]),
                             name='biases')

        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_size, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_size)),
                                name='weights'))

        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')

        logits = tf.matmul(hidden2, weights) + biases

    return logits

def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    loss_value = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    return loss_value

def training(loss_value, learning_rate):
    tf.summary.scalar('loss', loss_value)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss_value, global_step=global_step)

    return train_op

def fill_feed_dict(data_set, images_pl, label_pl):
    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict={
        images_pl: images_feed,
        label_pl: labels_feed
    }
    return feed_dict

def evaluation(logits, labels):
    eval_correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(eval_correct, tf.int32))

def do_eval(sess,
            eval_correct,
            images,
            labels,
            data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images, labels)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))

with tf.Graph().as_default():
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))

    label_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    sess = tf.InteractiveSession()

    logits = inference(images_placeholder, hidden1_size, hidden2_size)

    loss_value = loss(logits, label_placeholder)

    train_op = training(loss_value, learning_rate)

    eval_correct = evaluation(logits, label_placeholder)

    summary = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter('logs', sess.graph)

    tf.global_variables_initializer().run()

    for step in xrange(max_step):
        start_time = time.time()

        feed_dict = fill_feed_dict(data_sets.train, images_placeholder, label_placeholder)

        _, the_loss = sess.run([train_op, loss_value], feed_dict=feed_dict)

        duration = time.time() - start_time

        if step % 100 == 0:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, the_loss, duration))
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        if (step + 1) % 1000 == 0 or (step + 1)  == max_step:
            print('Training Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    label_placeholder,
                    data_sets.train)
            # Evaluate against the validation set.
            print('Validation Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    label_placeholder,
                    data_sets.validation)
            # Evaluate against the test set.
            print('Test Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    label_placeholder,
                    data_sets.test)

