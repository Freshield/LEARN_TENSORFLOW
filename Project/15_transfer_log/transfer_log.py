import numpy as np

filename = '0.9151_epoch99'
filename1 = 'Link_CNN_10k0.4020_epoch99'
test_str = '----------epoch 9 test accuracy is 0.851900----------'

accuracy = []
accuracy1 = []
context = ''
for line in open(filename):
    if 'test accuracy is' in line:
        index = line.find('0.')
        acc = round(float(line[index:index+5]), 2)
        accuracy.append(acc)
        context += str(acc) + ','

context += '\n'

for line in open(filename1):
    if 'test accuracy is' in line:
        index = line.find('0.')
        acc = round(float(line[index:index+5]), 2)
        accuracy.append(acc)
        context += str(acc) + ','

print accuracy
with open('accuracy1.csv', 'wb') as f:
    f.write(context)
