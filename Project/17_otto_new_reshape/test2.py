import numpy as np

#set num to one hot array
#ver 1.0
def num_to_one_hot(dataset, category_num):
    lines = dataset.shape[0]
    one_hot_dataset = np.zeros([lines, category_num], dtype=np.float32)
    one_hot_dataset[np.arange(lines), dataset] = 1
    return one_hot_dataset


def reshape_dataset(dataset):

    #100,94
    print dataset.shape

    #You need fill as your program
    x,y = dataset.shape

    feature_data = dataset[:,:-1]

    temp_data = np.zeros((x,y-1,y-1))
    input_data = np.zeros((x, y+2, y+2))

    for i in xrange(x):
        for j in xrange(y-1):
            right = feature_data[i, :j]
            left = feature_data[i, j:]
            temp_data[i, j] = np.concatenate((left, right))

    input_data[:,1:-2,1:-2] = temp_data[:,:,:]
    input_data = input_data.reshape((x,1,y+2,y+2))

    output_data = dataset[:, -1].astype(int)
    output_data = num_to_one_hot(output_data, 9)

    print input_data.shape

    return input_data, output_data

a = np.arange(8).reshape((2,4))
print a

input,output = reshape_dataset(a)

print input