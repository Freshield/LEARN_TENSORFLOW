import numpy as np

#set num to one hot array
#ver 1.0
def num_to_one_hot(dataset, category_num):
    lines = dataset.shape[0]
    one_hot_dataset = np.zeros([lines, category_num], dtype=np.float32)
    one_hot_dataset[np.arange(lines), dataset] = 1
    return one_hot_dataset

def reshape_dataset(dataset):

    #You need fill as your program]

    #feature data (x,35)
    feature_data = dataset[:,:-1]

    #temp data (x,35,35,35)
    temp_data = triple_size_data(feature_data)
    #x,y (batch,35)
    x, y, _, _ = temp_data.shape

    #input data (batch, 40,40,35)
    input_data = np.zeros((x, y+5, y+5, y))

    input_data[:,2:-3,2:-3,:] = temp_data[:,:,:]

    output_data = dataset[:, -1].astype(int)
    output_data = num_to_one_hot(output_data, 2)

    return input_data, output_data

def triple_size_data(dataset):
    #a_x is 2
    #a_y is 3
    a_x, a_y = dataset.shape

    output_dataset = np.zeros((a_x, a_y, a_y, a_y))

    for batch_num in range(a_x):

        # to create 3x3 matrix

        raw_data = np.zeros((a_y, a_y))

        for j in xrange(a_y):
            right = dataset[batch_num, :j]
            left = dataset[batch_num, j:]
            raw_data[j, :] = np.concatenate((left, right))

        #print raw_data

        # to create 3x3x3 matrix

        result = np.zeros((a_y, a_y, a_y))

        first = raw_data[0]
        #first is 123;231;312;
        #print first

        for x in range(a_y):
            temp_data = np.zeros((a_y, a_y))
            temp_data[:, :] = raw_data[:, :]
            #exchange xth and first value
            temp_data[x] = first
            temp_data[0] = raw_data[x]
            result[x, :, :] = temp_data[:, :]

        output_dataset[batch_num, :, :, :] = result[:, :, :]

    return output_dataset


a = np.arange(70).reshape((2,35))
b = np.hstack((a,[[0],[1]]))
print b
print

input = triple_size_data(a)

input,output = reshape_dataset(b)
print input.shape
print input[:,2:-3,2:-3,:]