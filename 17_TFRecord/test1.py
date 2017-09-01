import numpy as np

filename = '/media/freshield/BACKUP/FiberID_6fibers_20Spans_noPCA_1.csv'

count = 0
for line in open(filename, 'r'):
    print line[:10]
    count += 1
    if count == 50:
        break

print 'done'