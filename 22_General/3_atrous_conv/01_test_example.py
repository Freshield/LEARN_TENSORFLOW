#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: 01_test_example.py
@Time: 18-6-8 20:08
@Last_update: 18-6-8 20:08
@Desc: None
"""
import tensorflow as tf

img = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
img = tf.concat(values=[img,img],axis=3)

filter = tf.constant(value=1, shape=[3,3,2,5], dtype=tf.float32)
out_img = tf.nn.atrous_conv2d(value=img, filters=filter, rate=2, padding='SAME')

sess = tf.InteractiveSession()

print(out_img.eval())
print(out_img.eval().shape)