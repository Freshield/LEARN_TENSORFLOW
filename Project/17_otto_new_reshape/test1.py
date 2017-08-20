import numpy as np

a = np.array([1,2,3,4,5,6]).reshape((2,-1))

print a.shape

def reshape_dataset(dataset):
    a_x, a_y = dataset.shape

    input_data = np.zeros((a_x, a_y, a_y, a_y))

    for batch_num in range(a_x):

        # to create 3x3 matrix

        raw_data = np.zeros((a_y, a_y))

        for j in xrange(a_y):
            right = a[batch_num, :j]
            left = a[batch_num, j:]
            raw_data[j, :] = np.concatenate((left, right))

        print raw_data

        # to create 3x3x3 matrix

        result = np.zeros((a_y, a_y, a_y))

        first = raw_data[0]
        print first

        for x in range(a_y):
            temp_data = np.zeros((a_y, a_y))
            temp_data[:, :] = raw_data[:, :]
            temp_data[x] = first
            temp_data[0] = raw_data[x]
            result[x, :, :] = temp_data[:, :]

        input_data[batch_num, :, :, :] = result[:, :, :]

    print output
