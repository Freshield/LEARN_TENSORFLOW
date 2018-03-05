import pandas as pd
import numpy as np

def get_data_from_label_one_set(filename,batch_num):
    data = pd.read_csv(filename,header=None).values

    indexs = np.arange(data.shape[0])
    np.random.shuffle(indexs)

    ones_data = data[indexs[:batch_num]]
    return ones_data

filename = 'data/train_label_one_set.csv'

ones_data = get_data_from_label_one_set(filename,50)

print ones_data