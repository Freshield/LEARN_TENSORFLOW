import tensorflow as tf
from collections import namedtuple
from math import sqrt
from tensorflow.examples.tutorials.mnist import input_data


def conv2d(x, n_filters, k_h=5, k_w=5, stride_h=2, stride_w=2, stddev=0.02, activation=lambda x: x, bias=True,
           padding='SAME', name="Conv2D"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, x.get_shape()[-1], n_filters],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
        if bias:
            b = tf.get_variable('b', [n_filters], initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = conv + b
        return activation(conv)


def linear(x, n_units, scope=None, stddev=0.02, activation=lambda x: x):
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], n_units], tf.float32, tf.random_normal_initializer(stddev=stddev))
        return activation(tf.matmul(x, matrix))

    # ensure the path exist
def del_and_create_dir(dir_path):
    if tf.gfile.Exists(dir_path):
        tf.gfile.DeleteRecursively(dir_path)
    tf.gfile.MakeDirs(dir_path)


# %%
def residual_network(x, n_outputs, activation=tf.nn.relu):
    # %%
    LayerBlock = namedtuple('LayerBlock', ['num_repeats', 'num_filters', 'bottleneck_size'])
    blocks = [LayerBlock(3, 128, 32), LayerBlock(3, 256, 64), LayerBlock(3, 512, 128), LayerBlock(3, 1024, 256)]

    # %%
    input_shape = x.get_shape().as_list()
    if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        if ndim * ndim != input_shape[1]:
            raise ValueError('input_shape should be square')
        x = tf.reshape(x, [-1, ndim, ndim, 1])

    # %%
    # First convolution expands to 64 channels and downsamples
    net = conv2d(x, 64, k_h=7, k_w=7, name='conv1', activation=activation)

    # %%
    # Max pool and downsampling
    net = tf.nn.max_pool(net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # %%
    # Setup first chain of resnets
    net = conv2d(net, blocks[0].num_filters, k_h=1, k_w=1, stride_h=1, stride_w=1, padding='VALID', name='conv2')

    # %%
    # Loop through all res blocks
    for block_i, block in enumerate(blocks):
        for repeat_i in range(block.num_repeats):
            name = 'block_%d/repeat_%d' % (block_i, repeat_i)
            conv = conv2d(net, block.bottleneck_size, k_h=1, k_w=1, padding='VALID', stride_h=1, stride_w=1,
                          activation=activation, name=name + '/conv_in')

            conv = conv2d(conv, block.bottleneck_size, k_h=3, k_w=3, padding='SAME', stride_h=1, stride_w=1,
                          activation=activation, name=name + '/conv_bottleneck')

            conv = conv2d(conv, block.num_filters, k_h=1, k_w=1, padding='VALID', stride_h=1, stride_w=1,
                          activation=activation, name=name + '/conv_out')

            net = conv + net
        try:
            # upscale to the next block size
            next_block = blocks[block_i + 1]
            net = conv2d(net, next_block.num_filters, k_h=1, k_w=1, padding='SAME', stride_h=1, stride_w=1, bias=False,
                         name='block_%d/conv_upscale' % block_i)
        except IndexError:
            pass

    # %%
    net = tf.nn.avg_pool(net, ksize=[1, net.get_shape().as_list()[1], net.get_shape().as_list()[2], 1],
                         strides=[1, 1, 1, 1], padding='VALID')
    net = tf.reshape(net,
        [-1, net.get_shape().as_list()[1] * net.get_shape().as_list()[2] * net.get_shape().as_list()[3]])

    net = linear(net, n_outputs, activation=tf.nn.softmax)

    # %%
    return net


def rsnn():
    """Test the resnet on MNIST."""

    mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    y_pred = residual_network(x, 10)

    # %% Define loss/eval/training functions
    cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
    tf.summary.scalar('loss', cross_entropy)
    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

    # %% Monitor accuracy
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # define store path
    train_path = 'tmp/train'

    del_and_create_dir(train_path)

    # %% We now create a new session to actually perform the initialization the
    # variables:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # create writer for tensorboard
    train_writer = tf.summary.FileWriter(train_path, sess.graph)

    merged = tf.summary.merge_all()

    # %% We'll train in minibatches and report accuracy:
    batch_size = 50
    n_epochs = 5
    for epoch_i in range(n_epochs):
        # Training
        train_accuracy = 0
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            train_accuracy += sess.run([optimizer, accuracy], feed_dict={x: batch_xs, y: batch_ys})[1]
        train_accuracy /= (mnist.train.num_examples // batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys})
        train_writer.add_summary(summary, 0)

        # Validation
        valid_accuracy = 0
        for batch_i in range(mnist.validation.num_examples // batch_size):
            batch_xs, batch_ys = mnist.validation.next_batch(batch_size)
            valid_accuracy += sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        valid_accuracy /= (mnist.validation.num_examples // batch_size)
        print('epoch:', epoch_i, ', train:', train_accuracy, ', valid:', valid_accuracy)


if __name__ == '__main__':
    rsnn()