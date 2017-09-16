import numpy as np

filename = '0.7882_ls0.0192_epoch199'

context_train = ''
context_test = ''
for line in open(filename):
    if 'loss in loop 609 is' in line:
        line = line.split('acc')[1]
        index = line.find('.')
        index -= 1
        acc = round(float(line[index:index+5]), 3)
        context_train += str(acc) + ','
    if 'test acc in epoch' in line:
        index = line.find('0.')
        acc = round(float(line[index:index+5]), 3)
        context_test += str(acc) + ','

context = ''
context += context_train + '\n' + context_test


with open('accuracy_otto.csv', 'wb') as f:
    f.write(context)
