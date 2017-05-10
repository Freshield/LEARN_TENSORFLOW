import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping
import pickle
import sys
#set channels first
#K.set_image_dim_ordering('th')
print K.image_data_format()

#Load data from csv files

def create_output(output_1, output_2):

    return (output_1+1)*(output_2+1)-1

if len(sys.argv)>1:
    spans = sys.argv[1:]
    SPAN = []
    for s in spans:
        print 'SPAN ', s
        SPAN.append(int(s))
else:
    SPAN=[1]


NROWS = 1000 # for smaller datasets, choose from 100, 1000, 10000, and 'all'
#SPAN = [1]# choose the span from 1 to 20 for the prediciton

filename = '~/Ciena_data/ciena10000.csv'

data = pd.read_csv(filename, header=None, nrows=NROWS)

print "Data Shape: %s" % str(data.shape)

# Split to input and output
input_data = np.zeros((NROWS,31,100,3))


np_data = data.as_matrix()
temp_data = np.reshape(np_data[:,:6200], (NROWS,31,100,2))
input_data[:,:,:,0] = temp_data[:,:,:,0]
input_data[:,:,:,1] = temp_data[:,:,:,1]
input_data[:,:,:,2] = np.reshape(np.tile(np_data[:,6200:6241],76)[:,:3100],(NROWS,31,100))

# TODO instead of tiling, add 41 parameters in extra layer

if len(SPAN)==2:
    output_data = create_output(np_data[:,6240+SPAN[0]], np_data[:,6240+SPAN[1]])
else:
    print SPAN
    output_data = np_data[:,6240+SPAN[0]]
output_data = np_utils.to_categorical(output_data)

print input_data.shape
print output_data.shape

# normalize data

input_data[:,:,:,:2] = input_data[:,:,:,:2] - np.amin(input_data[:,:,:,:2])
input_data[:,:,:,:2] = input_data[:,:,:,:2]/np.amax(input_data[:,:,:,:2])
input_data[:,:,:,2] = input_data[:,:,:,2] - np.amin(input_data[:,:,:,2])
input_data[:,:,:,2] = input_data[:,:,:,2]/np.amax(input_data[:,:,:,2])

print 'Max of data is: ', np.amax(input_data)
print 'Min of data is: ', np.amin(input_data)

# Split into training and testing

X_train, X_test,y_train,y_test=train_test_split(input_data,output_data,test_size=0.15)

print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape

# max and min values of X training data

print np.amax(X_train)
print np.amin(X_train)
print X_train.dtype
print y_train.dtype


num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(31,100,3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 1000
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
# TODO try adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# training will stop once the validation accuraccy stops improving 0.1% over 5 consecutive epochs
early_stopping = EarlyStopping(monitor='val_acc', min_delta=.0001, patience = 10)

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, callbacks=[early_stopping])

# save metrics in a pickle file

if len(SPAN)==2:
    filename='results/log_span_%d_%d.pkl'%(SPAN[0],SPAN[1])
else:
    filename = 'results/log_span_%d.pkl' % (SPAN[0])

with open(filename,'wb') as handle:
    pickle.dump(history.history, handle)

# with open(filename, 'rb') as handle:
#     his = pickle.load(handle)
#
# print his

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.6f%%" % (scores[1]*100))
