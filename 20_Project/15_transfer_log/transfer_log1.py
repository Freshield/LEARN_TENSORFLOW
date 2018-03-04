import numpy as np

filename = '0.9151_epoch99'
test_str = '----------epoch 9 test accuracy is 0.851900----------'

context_train = ''
context_test = ''
for line in open(filename):
    if 'train acc in loop 599 is' in line:
        index = line.find('0.')
        acc = round(float(line[index:index+5]), 2)
        context_train += str(acc) + ','
    if 'test accuracy is' in line:
        index = line.find('0.')
        acc = round(float(line[index:index+5]), 2)
        context_test += str(acc) + ','

context = ''
context += context_train + '\n' + context_test


with open('accuracy2.csv', 'wb') as f:
    f.write(context)
