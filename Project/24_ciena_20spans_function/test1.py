import pandas as pd
import numpy as np
from calculate_min_max_model import *

file_amount = 1000

print 'Begin to split files'

total_min, total_max = create_min_max()

for filenum in range(file_amount):

    print filenum

    before_time = time.time()

    filename = dir_path + temp_name + str(filenum) + '.csv'

    minVal, maxVal = cal_minmax_split_single_file(filename, save_path)

    if filenum == 0:
        total_min = minVal
        total_max = maxVal
    else:
        # compare values
        total_min = mininum_values(total_min, minVal)
        total_max = maxium_values(total_max, maxVal)

    if filenum % 50 == 0:
        span_time = time.time() - before_time
        print "use %.2f second in 10 loop" % (span_time * 10)
        print "need %.2f minutes for all loop" % (((file_amount - filenum) * span_time) / 60)

array = np.zeros([2, 5], dtype=np.float32)
add_values_to_array(total_min, array, 0)
add_values_to_array(total_max, array, 1)
np.savetxt(save_path + 'minmax_value.csv', array, delimiter=",")
