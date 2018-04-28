import tensorflow as tf
import numpy as np

feature_columns = [tf.feature_column.numeric_column('x',shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

x_train = np.array([1,2,3,4],dtype=np.float32)
y_train = np.array([0,-1,-2,-3],dtype=np.float32)
x_eval = np.array([2,5,8,1],dtype=np.float32)
y_eval = np.array([-1.01,-4.1,-7,0])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x':x_train},y_train,batch_size=4,num_epochs=None,shuffle=True
)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x':x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False
)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x':x_eval},y_eval,batch_size=4,num_epochs=1000,shuffle=False
)

estimator.train(input_fn=input_fn,steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print('train metrics: %r' % train_metrics)
print('eval metrics: %r' % eval_metrics)