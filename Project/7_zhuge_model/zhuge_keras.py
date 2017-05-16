import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
import pandas as pd
import time
from zhuge_nn_model import *

NROWS = 10000

SPAN=[19]

log = ''

filename = '/home/freshield/Ciena_data/ciena_pca_10000.csv'

data = pd.read_csv(filename, header=None, nrows=NROWS)

words = 'Data Shape: %s' % str(data.shape)
print words
log += words + '\n'

train_set, validation_set, test_set = split_dataset(data, radio=0.1)

#normalize dataset
train_set, train_mean = normalize_dataset(train_set)
validation_set, _ = normalize_dataset(validation_set, train_mean)
test_set, _ = normalize_dataset(test_set, train_mean)

X_train, y_train = reshape_dataset(train_set, SPAN)
X_valid, y_valid = reshape_dataset(validation_set, SPAN)
X_test, y_test = reshape_dataset(test_set, SPAN)

print X_train.shape

#hypers
lr_rate = 0.002
max_step = 30000
batch_size = 100
lr_decay = 0.99
lr_epoch = 1000
act = 'relu'

model = Sequential()
model.add(Dense(50, input_dim=241, activation=act))
model.add(Dense(40, activation=act))
model.add(Dense(30, activation=act))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.002)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200,batch_size=128)
score = model.evaluate(X_test, y_test, batch_size=128)
print("Accuracy: %.6f%%" % (score[1]*100))