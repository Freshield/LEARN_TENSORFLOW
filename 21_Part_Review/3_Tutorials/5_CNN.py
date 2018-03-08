import tensorflow as tf
import re
import numpy as np

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

print()

average_grads = []
test = ([(1,2),(3,4),(5,6),([17,18],8)],[(9,10),(11,12),(13,14),([19,20],161)])
for grad_and_vars in zip(*test):
    print(1)
    print(grad_and_vars)
    j = [g for g, _ in grad_and_vars]
    print('j:',j)
    grad = np.stack(j, 0)
    print('grad:',grad)
    grad = np.mean(grad, 0)
    print('grad mean:',grad)
    print('j mean:',np.mean(j))
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)

print('average_grads')
print(average_grads)

test2 = [(1,2,1),(3,4,3),(5,6,5)]

a,b,c = zip(*test2)
print(a)
print(b)
print(c)

