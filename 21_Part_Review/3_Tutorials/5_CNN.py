import tensorflow as tf
import re

a = tf.constant([1,2,3])

b = tf.expand_dims(a,0)

print(a.shape)
print(b.shape)

sess = tf.InteractiveSession()

print(a.eval())

print(b.eval())

print(tf.__version__)

print(re.sub('%s_[0-9]*/' % 1, '', '1/hhh'))

info = [(1,4),(2,5),(3,6)]
a = [1,2,3]
b = [4,5,6]
c = [4,5,6,7,8]
zipped = zip(a,b)

zipped_list = list(zipped)

print(zipped_list)

unzip = zip(*zipped_list)
unzip_list = list(unzip)
print(unzip_list)
for i in unzip_list:
    print(i)