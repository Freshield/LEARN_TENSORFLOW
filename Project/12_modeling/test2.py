import numpy as np
import pandas as pd
import os

result = {
    1: lambda x,y: x + y

}

print result[1](2,2)

filename = '/home/freshield/Ciena_data/dataset_10k/ciena10000.csv'

print os.path.exists(filename)

def test():
    while True:
        print 'here'
        while True:
            print 'here1'
            return 0

test()