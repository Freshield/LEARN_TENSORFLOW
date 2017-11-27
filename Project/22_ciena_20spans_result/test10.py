import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import cycle


from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from itertools import cycle

import os
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

file_dir = '/media/freshield/COASAIR1/CIENA/Result/logs/all_20spans_predict_data/'

files = file_name(file_dir)

files = sorted(files, key=lambda d : int(d.split('pan')[-1].split('_')[0]))
print files

file_name_dic = {}
for i in range(20):
    file_name_dic[i+1] = files[i]

def get_one_hot(data, depth):
    a = tf.placeholder(dtype=tf.int32,shape=(None))

    b = tf.one_hot(a,depth=depth)

    with tf.Session() as sess:
        feed_dict = {a: data}
        result = sess.run(b,feed_dict=feed_dict)
        return result

file_dir = '/media/freshield/COASAIR1/CIENA/Result/logs/all_20spans_predict_data/'
#file_name = 'span1_result_acc_0.9948.csv'
#file_name = 'span2_result_acc_0.9755.csv'
#file_name = 'span3_result_acc_0.9555.csv'
file_name = file_name_dic[20]
print file_name

data = pd.read_csv(file_dir+file_name).values

temp = data[:,0].astype(np.int32)
y_score = get_one_hot(temp,6)
temp = data[:,1].astype(np.int32)
y_test = get_one_hot(temp, 6)

n_classes = 6

#for each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:,i],y_score[:,i])

#plot pr curve for each class
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','blue','green','red','cyan','magenta','yellow'])

plt.figure(figsize=(7,8))
f_scores = np.linspace(0.2,0.8,num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y>=0], y[y>=0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9,y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('precision-recall for class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('extension of precision-recall curve to multi-class')
plt.legend(lines,labels,loc=(0,-.38),prop=dict(size=14))

plt.show()