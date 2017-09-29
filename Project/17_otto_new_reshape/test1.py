import numpy as np

a = np.array([1,2,3,4,5,6]).reshape((2,-1))

print a.shape

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

output = triple_size_data(a)
print output

b = np.arange(15).reshape((3,5))

output = triple_size_data(b)
print output.shape
