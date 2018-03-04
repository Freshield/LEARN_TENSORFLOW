from file_system_model import *
from basic_model import *
import flow_model as fm
import Resnet_link_model as rl
import pandas as pd
import numpy as np
import os
import time


########################################################
#                 HELPER                               #
########################################################
def normalize_dataset(dataset, min_values, max_values):
    norm_dataset = np.zeros((dataset.shape))
    norm_dataset[:, :] = dataset[:, :]

    CMr_min, CMi_min, CD_min, length_min, power_min = min_values

    CMr_max, CMi_max, CD_max, length_max, power_max = max_values

    def calcul_norm(dataset, min, max):
        return (2 * dataset - max - min) / (max - min)

    norm_dataset[:, 0:12000] = calcul_norm(norm_dataset[:, 0:12000], CMr_min, CMr_max)
    norm_dataset[:, 12000:24000] = calcul_norm(norm_dataset[:, 12000:24000], CMi_min, CMi_max)
    norm_dataset[:, 24000:24001] = calcul_norm(norm_dataset[:, 24000:24001], CD_min, CD_max)
    norm_dataset[:, 24001:24021] = calcul_norm(norm_dataset[:, 24001:24021], length_min, length_max)
    norm_dataset[:, 24021:24041] = calcul_norm(norm_dataset[:, 24021:24041], power_min, power_max)

    return norm_dataset


########################################################

# restore the model
# ver 1.0
def predict_type_enlc(dataset, model_path=None):
    print 'Begin to run predict'
    begin_time = time.time()

    if model_path == None:
        model_path = 'module/module.ckpt'

    model = rl

    dpm.model = model

    min_value = (-0.040546, -0.098555997, 2563.8999, -26.066, -10.398)
    max_value = (0.041134998, 0.023029, 25980.0, 193.57001, 22.183001)

    norm_dataset = normalize_dataset(dataset, min_value, max_value)

    norm_time = time.time()
    print 'norm dataset cost %.3f seconds for %d lines data' % (norm_time - begin_time, dataset.shape[0])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            with tf.device('/cpu:0'):
                sess_time = time.time()
                print 'create session cost %.3f seconds' % (sess_time - norm_time)

                input_x = tf.placeholder(tf.float32, [None, 304, 48, 2], name='input_x')
                para_pl = tf.placeholder(tf.float32, [None, 41], name='para_pl')
                train_phase = tf.placeholder(tf.bool, name='train_phase')
                keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                ENLC_array = tf.reshape(tf.constant([34.515, 23.92, 21.591, 25.829, 28.012, 29.765], dtype=tf.float32),
                                        [6, 1])

                # logits
                y_pred, parameters = model.inference(input_x, para_pl, train_phase, keep_prob)

                y_prob = tf.nn.softmax(y_pred)

                y_enlc = tf.matmul(y_prob, ENLC_array)

                y_type = tf.argmax(y_pred, 1)

                model_time = time.time()
                print 'create model cost %.3f seconds' % (model_time - sess_time)

                saver = tf.train.Saver()

                saver.restore(sess, model_path)

                load_time = time.time()
                print 'load model cost %.3f seconds' % (load_time - model_time)

                X_test, para_test = model.reshape_test_dataset(norm_dataset)
                reshape_time = time.time()
                print 'reshape data cost %.3f seconds for %d lines data' % (reshape_time - load_time, dataset.shape[0])

                feed_dict = {input_x: X_test, para_pl: para_test, train_phase: False, keep_prob: 1.0}

                y_type_v, y_enlc_v = sess.run([y_type, y_enlc], feed_dict=feed_dict)

                predict_time = time.time()
                print 'run the predict cost %.3f seconds for %d lines data' % (predict_time - reshape_time,
                                                                               dataset.shape[0])

                print 'total predict cost %.3f seconds for %d lines data' % (
                predict_time - begin_time, dataset.shape[0])

    return y_type_v, y_enlc_v


filename = 'data_sample.csv'

data = pd.read_csv(filename, header=None).values[:100]

type_v, enlc_v = predict_type_enlc(data)
