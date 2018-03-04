import numpy as np

filename = '0.9245_epoch76'

context_train = ''
context_test = ''
for line in open(filename):
    if 'train acc in loop 1000 is' in line:
        index = line.find('0.')
        acc = round(float(line[index:index+5]), 3)
        context_train += str(acc) + ','
    if 'test accuracy is' in line:
        index = line.find('0.')
        acc = round(float(line[index:index+5]), 3)
        context_test += str(acc) + ','

context = ''
context += context_train + '\n' + context_test


with open('accuracy_ciena_20spans.csv', 'wb') as f:
    f.write(context)
