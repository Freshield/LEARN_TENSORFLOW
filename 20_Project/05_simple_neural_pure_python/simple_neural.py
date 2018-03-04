import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.solver import Solver
import pandas as pd

def split_dataset(dataset, radio):
    test_dataset_size = int(radio * len(dataset))
    data_size = dataset.shape[0]
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    train_set = dataset.values[indexs[:-test_dataset_size * 2]]
    validation_set = dataset.values[indexs[-test_dataset_size * 2 : -test_dataset_size]]
    test_set = dataset.values[indexs[-test_dataset_size : ]]
    return train_set, validation_set, test_set

def get_whole_data(data_set):
    features = data_set[:,:241]
    labels = data_set[:,259]
    return {'features':features.astype(np.float64), 'labels':labels.astype(np.int64)}
#-----------------------------
filename = 'ciena_test.csv'
dataset = pd.read_csv(filename, header=None)
train_dataset, validation_dataset, test_dataset = split_dataset(dataset, 0.1)

train_data = get_whole_data(train_dataset)
validation_data = get_whole_data(validation_dataset)

data = {'X_train':train_data['features'], 'y_train':train_data['labels'],
        'X_val':validation_data['features'], 'y_val':validation_data['labels']}

hidden_dims = [512, 256, 128]
reg = 0.01
lr = 0.002
weight_scale = 0.1
num_epoch = 500

model = FullyConnectedNet(hidden_dims=hidden_dims, input_dim=241, num_classes=3,
                          dtype=np.float64, reg=reg, weight_scale=weight_scale,
                          use_batchnorm=True, dropout=0.0)

solver = Solver(model, data, num_epochs=num_epoch, batch_size=100, update_rule='adam',
                optim_config={
                    'learning_rate':lr
                },
                print_every=500, lr_decay=0.99, verbose=True)

solver.train()
