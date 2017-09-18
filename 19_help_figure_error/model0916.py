# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 14:41:31 2017

@author: Linstancy
"""

import tensorflow as tf 
import numpy as np
import preprocessing as prepro
import cv2
import csv
import os
import skimage.data
import skimage.transform
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

train_data_dir = os.path.normpath("GTSRB/Final_Training/Images")
test_data_dir = os.path.normpath("GTSRB/Final_Test/Images")

def load_data(data_dir):
    """Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    y_train = []
    x_train = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            x_train.append(skimage.data.imread(f))
            y_train.append(int(d))
    
    return x_train, y_train

    y_test = []
    x_test = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            x_test.append(skimage.data.imread(f))
            y_test.append(int(d))
    return y_test, x_test

def lsy_model(x, train):
    
    with tf.name_scope('conv_1'):
        w1=tf.Variable(tf.truncated_normal(shape=[1,1,3,32]))
        b1=tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32))
        conv_1=tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME')+b1
        relu1=tf.nn.relu(conv_1)
        
    with tf.name_scope('conv_2'):
        w2=tf.Variable(tf.truncated_normal(shape=[3,3,32,64]))
        b2=tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32))
        conv_2=tf.nn.conv2d(relu1,w2,strides=[1,1,1,1],padding='SAME')+b2
        pool2=tf.nn.max_pool(conv_2,[1,2,2,1],[1,2,2,1],padding='SAME')
        relu2=tf.nn.relu(pool2)
    
    with tf.name_scope('conv_3'):
        w3=tf.Variable(tf.truncated_normal(shape=[3,3,64,192]))
        b3=tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32))
        conv_3=tf.nn.conv2d(relu2,w3,strides=[1,1,1,1],padding='SAME')+b3
        pool3=tf.nn.max_pool(conv_3,[1,2,2,1],[1,2,2,1],padding='SAME')
        relu3=tf.nn.relu(pool3)
        
    with tf.name_scope('conv4'):
        w4=tf.Variable(tf.truncated_normal(shape=[3,3,192,192]))
        b4=tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32))
        conv_4=tf.nn.conv2d(relu3,w4,strides=[1,1,1,1],padding='SAME')+b4
        pool4=tf.nn.max_pool(conv_4,[1,2,2,1],[1,2,2,1],padding='SAME')
        relu4=tf.nn.relu(pool4)
        
    fc0=flatten(relu4)
    
    with tf.name_scope('fc1'):
        w5=tf.Variable(tf.truncated_normal(shape=[6912,2048]))
        b5=tf.Variable(tf.constant(0.0, shape=[2048], dtype=tf.float32))
        fc1=tf.matmul(fc0,w5)+b5
        fc1=tf.nn.relu(fc1)
        if train: fc1=tf.nn.dropout(fc1,0.5)
    
    with tf.name_scope('fc2'):
        w6=tf.Variable(tf.truncated_normal(shape=[2048,1024]))
        b6=tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32))
        fc2=tf.matmul(fc1,w6)+b6
        fc2=tf.nn.relu(fc2)
        if train: fc2=tf.nn.dropout(fc2,0.5)
    
    with tf.name_scope('fc3'):
        w7=tf.Variable(tf.truncated_normal(shape=[1024,256]))
        b7=tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32))
        fc3=tf.matmul(fc2,w7)+b7
        fc3=tf.nn.relu(fc3)
        if train: fc2=tf.nn.dropout(fc3,0.5)
        
    with tf.name_scope('fc4'):
        w8=tf.Variable(tf.truncated_normal(shape=[256,43]))
        b8=tf.Variable(tf.constant(0.0, shape=[43], dtype=tf.float32))
        logit=tf.matmul(fc3,w8)+b8
        
        if train: logit=tf.nn.dropout(logit,0.5)
        
        
    return logit

x_train,y_train=load_data(train_data_dir)
x_train= np.array([skimage.transform.resize(image, (48, 48)) for image in x_train])


x_test,y_test=load_data(test_data_dir)
x_test= np.array([skimage.transform.resize(image, (48, 48)) for image in x_test])

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 48, 48, 3], name='x-input')
    y = tf.placeholder(tf.int32, [None,43],name='y-input')
    one_hot_y = tf.one_hot(y, 43, name='y-one-hot')

rate = 0.001
global_step = tf.placeholder(tf.int32)
logit=lsy_model(x,train=True)
EPOCHS=1000
BATCH_SIZE=100
with tf.name_scope('optimizer'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    learning_rate = tf.train.exponential_decay(rate, global_step, 1, 0.99, staircase=True, name=None)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_operation = optimizer.minimize(loss_operation)
    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(one_hot_y, 1), tf.argmax(logits, 1), num_classes)        

with tf.Session() as sess:
    
    tf.initialize_all_variables().run()
    print("Training...")
    for i in range(EPOCHS):
        x_train_epoch, y_train_epoch = shuffle(x_train, y_train) 
        num_examples = len(x_train)
        #start_time = time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x,batch_y= x_train_epoch[offset:end],y_train_epoch[offset:end]
            _, loss, learning_rate_val = sess.run([loss_operation,training_operation,learning_rate], feed_dict={x:batch_x,y:batch_y})
            
            
        '''if i % 100 == 0 or i + 1 == EPOCHS:
            valid_acc=sess.run(accuracy_operation,feed_dict=valid_feed)
            print(" %d training step(s),validation accuracy is %g" % (i,valid_acc))'''
        
        test_acc=sess.run(accuracy_operation,feed_dict={x:x_test, y:y_test})
        print("%d training step(s), test accuracy is %g" % (i, test_acc))
        
    
    
    
    
        
        
    
    
        
        
                           












































