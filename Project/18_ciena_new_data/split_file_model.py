import pandas as pd
import numpy as np

for filenum in [3,4,5,6,7,8,9,10]:

    dir_name = '/media/freshield/DATA_U/Ciena_data/new_data/FiberID_6fibers_10Spans/'
    filename = 'FiberID_6fibers_10Spans_noPCA_'
    #filenum = '3.csv'
    dataSize = 100 * 1000

    reader = pd.read_csv(dir_name + filename + str(filenum) + '.csv', header=None, iterator=True, dtype=np.float32)

    chunkSize = 1000

    loop = True

    count = 0

    while loop:
        try:

            chunk = reader.get_chunk(chunkSize)
            print count
            print count + (filenum - 1) * 100
            np.savetxt(dir_name + 'RecutFile/Raw_data_' + str(count + (filenum - 1) * 100) + '.csv', chunk,
                       delimiter=',')

            count += 1


        except StopIteration:
            print "stop"
            break
