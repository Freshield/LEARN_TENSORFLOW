import numpy as np


def random_uniform_array(number, start, end):
    array = np.zeros(number)
    for i in np.arange(number):
        array[i] = 10 ** np.random.uniform(start, end)

    return array


a = random_uniform_array(10, -0.3, 0)

print a