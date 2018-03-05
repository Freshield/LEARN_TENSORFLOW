import pandas as pd
import numpy as np

filename = 'data/sample.csv'

datasize = 6

reader = pd.read_csv(filename, iterator=True)

loop = True
chunkSize = 2
count = 0

total_loop = datasize // chunkSize

rest_data = datasize % chunkSize


print total_loop
print rest_data

print 'begin to calculate the min max value'
while loop:

    try:

        print count
        chunk = reader.get_chunk(chunkSize)
        print 'read chunk'
        print chunk.values[:,-3:]
        print

        count += 1


    except StopIteration:
        print "stop"
        break
