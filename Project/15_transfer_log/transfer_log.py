import numpy as np

filename = '0.9151_epoch99'
test_str = '----------epoch 9 test accuracy is 0.851900----------'

accuracy = []
context = ''
for line in open(filename):
    if 'test accuracy is' in line:
        index = line.find('0.')
        acc = round(float(line[index:index+5]), 2)
        accuracy.append(acc)
        context += str(acc) + ','

print accuracy
with open('accuracy.csv', 'wb') as f:
    f.write(context)
